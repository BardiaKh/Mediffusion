from .diffusion.base import GaussianDiffusionBase, ModelMeanType, ModelVarType, LossType
from .diffusion.solvers import DDPMSolver, DDIMSolver, InverseDDIMSolver, PNMDSolver
from .utils.diffusion import get_named_beta_schedule, get_respaced_betas, enforce_zero_terminal_snr, UniformSampler
from .utils.pl import get_obj_from_str
from .models.unet import UNetModel
import os
import torch
import numpy as np
import bkh_pytorch_utils as bpu
from tqdm import tqdm
import torchextractor as tx
from functools import partial
from omegaconf import OmegaConf

class DiffusionModule(bpu.BKhModule):
    def __init__(
        self,
        config_file,
        train_ds=None,
        val_ds=None,
        collate_fn=None,
        val_collate_fn=None,
        train_sampler=None,
        val_sampler=None,
        dl_workers=None,
        batch_size=None,
        val_batch_size=None,
        **kwargs
    ):
        super().__init__(
            collate_fn=collate_fn, 
            val_collate_fn=val_collate_fn, 
            train_sampler=train_sampler, 
            val_sampler=val_sampler, 
            train_ds=train_ds, 
            val_ds=val_ds, 
            dl_workers=dl_workers, 
            batch_size=batch_size, 
            val_batch_size=val_batch_size
        )

        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        default_config_file = os.path.join(current_directory, 'default_config', 'default.yaml')
        default_config = OmegaConf.load(default_config_file)
        user_config = OmegaConf.load(config_file)
        self.config = OmegaConf.merge(default_config, user_config)

        # add batch size to config just for logging purposes
        OmegaConf.update(self.config, "batch_size", self.batch_size, merge=False)
        
        self.lr = self.config.optimizer.lr
        self.optimizer_class = get_obj_from_str(self.config.optimizer.type)

        initial_betas = get_named_beta_schedule(
            schedule_name= self.config.diffusion.schedule_name,
            num_diffusion_timesteps= self.config.diffusion.timesteps,
            **self.config.diffusion.schedule_params
        )
        
        if self.config.diffusion.timestep_respacing is None:
            timestep_respacing = [self.config.diffusion.timesteps]
        
        final_betas, self.timestep_map  = get_respaced_betas(initial_betas, timestep_respacing)
        if self.config.diffusion.enforce_zero_terminal_snr:
            final_betas = enforce_zero_terminal_snr(final_betas)
        
        self.diffusion = GaussianDiffusionBase(
            betas = final_betas,
            model_mean_type = ModelMeanType[self.config.diffusion.mean_type],
            model_var_type = ModelVarType[self.config.diffusion.var_type],
            loss_type = LossType[self.config.diffusion.loss_type],
        )
        
        self.model_input_shape = (
            self.config.model.in_channels - self.config.model.concat_channels,
            *([self.config.model.input_size]*self.config.model.dims if isinstance(self.config.model.input_size, int) else self.config.model.input_size),
        )
        
        self.model = UNetModel(**self.config.model)

        self.timestep_scheduler = UniformSampler(self.diffusion.num_timesteps, self.timestep_map)

        self.class_conditioned = False if self.config.model.num_classes == 0 else True
        self.concat_conditioned = False if self.config.model.concat_channels == 0  else True
        
        if not self.class_conditioned:
            OmegaConf.update(self.config, "validation.classifier_cond_scale", 0, merge=False)

        self.save_hyperparameters(self.config)
        
    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

    def setup_feature_extractor(self, steps, blocks):
        block_names = []
        for block in blocks:
            block_names.append(f"output_blocks.{block}")
        self.feature_extractor = tx.Extractor(self.model, block_names)
        self.feature_extractor_steps = sorted(steps)
        
    def unset_feature_extractor(self):
        self.feature_extractor = None
        self.feature_extractor_steps = None

    @torch.inference_mode()
    def get_cls_embedding(self, cls):
        return self.model.class_embed(cls.to(device=self.device, dtype=self.dtype))

    @torch.inference_mode()
    def forward_features(self, x_start, model_kwargs = None, noise=None, return_dict=True):
        assert self.feature_extractor_steps is not None, "Feature extractor is not setup, run setup_feature_extractor() first!"

        x_start = x_start.to(self.device)
        if len(x_start.shape) == len(self.model_input_shape): 
            x_start = x_start.unsqueeze(0)
        else:
            assert len(x_start.shape) == len(self.model_input_shape) + 1, f"Input shape is not compatible with model input shape ({self.model_input_shape})"
        
        upsampler = partial(torch.nn.functional.interpolate, size=self.model_input_shape[-2], mode="nearest-exact")
        all_features = {}
        for step in self.feature_extractor_steps:
            t = self.timestep_map.gather(-1, torch.tensor([step]).long()).to(self.device)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise).to(dtype=x_start.dtype)
            _, features = self.feature_extractor(x_t, t, **model_kwargs)

            all_features[step] = {}
            
            for key in features:
                all_features[step][key] = upsampler(features[key])

        if return_dict:
            return all_features
        else:
            final_feature_tensor = []
            for i, step in enumerate(self.feature_extractor_steps):
                tensor = torch.cat([all_features[step][key] for i in all_features[step]], dim=1)
                final_feature_tensor.append(tensor)
            
            final_feature_tensor = torch.cat(final_feature_tensor, dim=1)
            return final_feature_tensor

    def training_step(self, batch, batch_idx):
        x_start = batch["img"]
        cls = batch["cls"] if self.class_conditioned else None
        concat = batch['concat'] if self.concat_conditioned else None

        batch_size = x_start.shape[0]
        ts, t_weights = self.timestep_scheduler.sample(batch_size=batch_size, device=x_start.device)
        noise = torch.randn_like(x_start).to(device = self.device, dtype=x_start.dtype)
        
        model_kwargs = {"cls": cls, "concat": concat}

        loss_terms = self.diffusion.training_losses(model=self.model, x_start=x_start, t=ts, noise=noise, model_kwargs=model_kwargs)

        self.log(f'train_loss', loss_terms['loss'].mean(), on_epoch=True, on_step=True, prog_bar=False)

        return loss_terms['loss'].mean()

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        x_start = batch["img"]
        cls = batch["cls"] if self.class_conditioned else None
        concat = batch['concat'] if self.concat_conditioned else None

        batch_size = x_start.shape[0]
        model_kwargs = {"cls": cls, "concat": concat}
        noise = torch.randn_like(x_start).to(device=self.device, dtype=x_start.dtype)

        if batch_idx == 0:
            assert len(x_start.shape)==4 or (len(x_start.shape)==5 and batch_size==1), f"expected 2D image (with any batch size) or 3D image (with batch size 1), got {x_start.shape}"

            imgs = self.predict(noise, inference_protocol=self.config.validation.protocol, model_kwargs=model_kwargs, classifier_cond_scale=self.config.validation.classifier_cond_scale)
            
            if self.config.validation.log_original:
                x_start_list = x_start.cpu().split(1, dim=0)
                x_start_list = [img.squeeze(0) for img in x_start_list]
                self._log_img(x_start_list, cls, title = "real samples")
            
            self._log_img(imgs, cls, title = "generated samples")
        
        ts, t_weights = self.timestep_scheduler.sample(batch_size=batch_size, device=x_start.device)
        
        loss_terms = self.diffusion.training_losses(model=self.model, x_start=x_start, t=ts, noise=noise, model_kwargs=model_kwargs)

        self.log(f'val_loss', loss_terms['loss'].mean(), on_epoch=True, on_step=False, prog_bar=False)

        return {'val_loss': loss_terms['loss'].mean()}

    def _log_img(self, imgs, cls, title="generated samples"):
        if self.config.model.dims == 3:
            imgs = torch.stack(imgs, dim=0)                     # (B, C, H, W, D)
            imgs = imgs[:,0:1,:,:,:]                            # C:1
            imgs = imgs.permute(0,4,1,2,3)                      # (B, D, C, H, W)
            if self.class_conditioned:
                cls = cls.repeat(imgs.shape[1], 1)              # (B*D, C)
            imgs = imgs.reshape(-1, *tuple(imgs.shape[2:]))     # (B*D, C, H, W)
            imgs = imgs.split(1, dim=0)                         # [(1, C, H, W)] * B*D
            imgs = [img.squeeze(0) for img in imgs]             # [(C, H, W)] * B*D
        
        imgs_to_log = []
        for i,img in enumerate(imgs):
            img = img.squeeze(0).numpy()
            img = img.transpose(1,0)
            imgs_to_log.append(img)

        if self.class_conditioned:
            caption = []
            cls = cls.cpu().split(1, dim=0)
            for i,c in enumerate(cls):
                c = c.numpy().tolist()
                caption.append(f"Class: {c}")
            self.logger.log_image(key=title, images=imgs_to_log, caption=caption)
        else:
            self.logger.log_image(key=title, images=imgs_to_log)

    @torch.inference_mode()
    def predict(self, init_noise, inference_protocol="DDPM", model_kwargs=None, classifier_cond_scale=None, generator=None, start_denoise_step=None, post_process_fn=None, clip_denoised=True):            
        init_noise = init_noise.to(device=self.device, dtype=self.dtype)

        for key in model_kwargs:
            if model_kwargs[key] is not None:
                model_kwargs[key] = model_kwargs[key].to(device=self.device, dtype=self.dtype)
        
        if inference_protocol == "DDPM":
            solver = DDPMSolver(self.diffusion)
        elif inference_protocol.startswith("DDIM"):
            num_steps = int(inference_protocol[len("DDIM"):])
            solver = DDIMSolver(self.diffusion, num_steps=num_steps)
        elif inference_protocol.startswith("IDDIM"):
            num_steps = int(inference_protocol[len("IDDIM"):])
            solver = InverseDDIMSolver(self.diffusion, num_steps=num_steps)
        elif inference_protocol.startswith("PNMD"):
            num_steps = int(inference_protocol[len("PNMD"):])
            solver = PNMDSolver(self.diffusion, num_steps=num_steps)
        else:
            raise ValueError(f"Unknown inference protocol {inference_protocol}, only DDPM, DDIM, IDDIM, PNMD are supported")

        imgs = solver.sample(
            self.model,
            init_noise,
            start_denoise_step=start_denoise_step,
            cond_scale=classifier_cond_scale,
            model_kwargs=model_kwargs,
            clip_denoised=clip_denoised,
            generator=generator
        )
        
        imgs = imgs.split(1, dim=0)                 # [(1, C, H, W, (D))] * B
        imgs = [img.squeeze(0) for img in imgs]     # [(C, H, W, (D))] * B

        if not inference_protocol.startswith("IDDIM") and post_process_fn is not None:
            for i in range(len(imgs)):
                img = imgs[i]
                img = post_process_fn(img)
                imgs[i] = img
                
        for i in range(len(imgs)):
            imgs[i] = imgs[i].cpu()

        return imgs # [C, H, W, (D)] * B

    def configure_optimizers(self):
        if self.total_steps is None:
            max_epochs=self.trainer.max_epochs
            grad_acc=self.trainer.accumulate_grad_batches
            self.set_total_steps(steps=len(self.train_dataloader())*max_epochs//grad_acc)

        params = bpu.add_weight_decay(self.model,5e-4)
        optimizer = self.optimizer_class(params, lr=self.lr) 
        return optimizer