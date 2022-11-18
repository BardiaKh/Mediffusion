from diffusion_base import GaussianDiffusionBase, SpacedDiffusion, ModelMeanType, ModelVarType, LossType
import pytorch_lightning as pl
from .utils.diffusion import get_named_beta_schedule, space_timesteps, UniformSampler
from unet import UNetModel, SuperResModel
import torch
import numpy as np
import yaml
import bkh_pytorch_utils as bpu
from tqdm import tqdm
from nn import mean_flat
import torchextractor as tx
from functools import partial

from skimage.io import imsave

class DiffusionPLModule(bpu.BKhModule):
    def __init__(
        self,
        num_diffusion_timesteps,
        beta_schedule_name="cosine", beta_start=1e-4, beta_end=2e-2, cosine_s=0.008,
        task_type = "unsupervised",
        timestep_scheduler_name="uniform",
        timestep_respacing="",
        model_config_path="configs/model_unsupervised.yaml",
        model_mean_type = ModelMeanType.EPSILON,
        model_var_type = ModelVarType.FIXED_SMALL,
        loss_type = LossType.MSE,
        classifier_cond_scale=0,
        inference_protocol="DDPM",
        input_size=256,
        collate_fn=None, 
        val_collate_fn=None,
        train_sampler=None,val_sampler=None, ddp_sampler=False, 
        train_ds=None, val_ds=None, dl_workers=-1,
        batch_size=16, val_batch_size=None, lr=1e-4,
    ):
        super().__init__(collate_fn=collate_fn, val_collate_fn=val_collate_fn, train_sampler=train_sampler, val_sampler=val_sampler, ddp_sampler=ddp_sampler, train_ds=train_ds, val_ds=val_ds, dl_workers=dl_workers, batch_size=batch_size, val_batch_size=val_batch_size)
        self.task_type = task_type
        self.loss_type = loss_type
        self.classifier_cond_scale = classifier_cond_scale
        self.inference_protocol = inference_protocol
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.lr = lr

        self.initial_betas = get_named_beta_schedule(beta_schedule_name,num_diffusion_timesteps, beta_start, beta_end, cosine_s)
        if timestep_respacing=="":
            timestep_respacing = [num_diffusion_timesteps]
        
        final_betas, self.timestep_map  = self.get_respaced_betas(self.initial_betas, timestep_respacing)
        
        self.diffusion = GaussianDiffusionBase(final_betas, model_mean_type, model_var_type, loss_type)

        model_config = self.get_model_config(model_config_path, input_size)

        self.model_input_shape = (model_config["in_channels"], *[model_config["image_size"]]*model_config['dims']) # excluding batch dimension

        if self.task_type == "unsupervised":
            self.model = UNetModel(**model_config)
        elif self.task_type.startswith("superres"):
            self.model = SuperResModel(**model_config)

        if timestep_scheduler_name == "uniform":
            self.timestep_scheduler = UniformSampler(self.diffusion.num_timesteps, self.timestep_map)
        else:
            raise NotImplemented("Only uniform timestep scheduler is supported for now")

        self.class_conditioned = False if model_config['num_classes']==0 else True

        if self.inference_protocol == "DDPM":
            self.inference_diffusion = self.diffusion
        elif self.inference_protocol.startswith("DDIM"):
            _, timestep_map = self.get_respaced_betas(self.initial_betas, self.inference_protocol)
            self.inference_diffusion = SpacedDiffusion(use_timesteps=timestep_map, betas=self.initial_betas, model_mean_type=self.model_mean_type, model_var_type=self.model_var_type, loss_type=self.loss_type, rescale_timesteps=None)
        else:
            raise ValueError(f"Unknown inference protocol {self.inference_protocol}, only DDPM, DDIM are supported")

        # self.save_hyperparameters()
        
    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

    def get_respaced_betas(self, initial_betas, timestep_respacing):
        use_timesteps = space_timesteps(len(initial_betas), timestep_respacing)
        alphas = 1.0 - initial_betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        last_alpha_cumprod = 1.0
        new_betas = []
        timestep_map = []
        for i, alpha_cumprod in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                timestep_map.append(i)

        new_betas = torch.tensor(new_betas)
        timestep_map = torch.tensor(timestep_map)
        return new_betas, timestep_map

    def get_model_config(self, config_path, input_size):
        model_config = yaml.safe_load(open(config_path, "r"))
        model_config["image_size"] = input_size
        
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            model_config["out_channels"] = 2 * model_config["in_channels"]
        else:
            model_config["out_channels"] = model_config["in_channels"]

        if model_config['channel_mult'] == "":
            if model_config["image_size"] == 512:
                model_config['channel_mult'] = (0.5, 1, 1, 2, 2, 4, 4)
            elif model_config["image_size"] == 256:
                model_config['channel_mult'] = (1, 1, 2, 2, 4, 4)
            elif model_config["image_size"] == 128:
                model_config['channel_mult'] = (1, 1, 2, 3, 4)
            elif model_config["image_size"] == 64:
                model_config['channel_mult'] = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {model_config['image_size']}")
        else:
            model_config['channel_mult'] = tuple(int(ch_mult) for ch_mult in model_config['channel_mult'].split(","))

        attention_ds = []
        for res in model_config['attention_resolutions'].split(","):
            attention_ds.append(input_size // int(res))

        model_config['attention_resolutions'] = attention_ds

        model_config['num_res_blocks'] = [int(i) for i in str(model_config['num_res_blocks']).split(",")]
        
        return model_config

    def setup_feature_extractor(self, steps, blocks):
        assert self.task_type == "unsupervised", "Feature extractor is only supported for unsupervised tasks"
        block_names = []
        for block in blocks:
            block_names.append(f"output_blocks.{block}")
        self.feature_extractor = tx.Extractor(self.model, block_names)
        self.feature_extractor_steps = sorted(steps)


    @torch.no_grad()
    def forward_features(self, x_start, noise=None, return_dict=True):
        assert self.task_type == "unsupervised", "Feature extractor is only supported for unsupervised tasks"
        assert self.feature_extractor_steps is not None, "Feature extractor is not setup, run setup_feature_extractor() first!"

        x_start = x_start.to(self.device)
        if len(x_start.shape) == len(self.model_input_shape): 
            x_start = x_start.unsqueeze(0)
        else:
            assert len(x_start.shape) == len(self.model_input_shape) + 1, f"Input shape is not compatible with model input shape ({self.model_input_shape})"
        
        upsampler = partial(torch.nn.functional.interpolate, size=self.model_input_shape[-2], mode="nearest")
        all_features = {}
        for step in self.feature_extractor_steps:
            t = self.timestep_map.gather(-1, torch.tensor([step]).long()).to(self.device)
            x_t = self.diffusion.q_sample(x_start, t, noise=noise).to(dtype=x_start.dtype)
            model_output, features = self.feature_extractor(x_t, t)

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
        # ported from train_losses in diffusion.py
        x_start = batch["img"]
        cls = batch["cls"] if self.class_conditioned else None
        batch_size = x_start.shape[0]
        ts, t_weights = self.timestep_scheduler.sample(batch_size=batch_size, device=x_start.device)
        noise = torch.randn_like(x_start).to(device = self.device, dtype=x_start.dtype)
        
        model_kwargs = {"cls": cls}
        if self.task_type.startswith("superres"):
            low_res_img = batch["low_res_img"]
            model_kwargs["low_res"] = low_res_img

        loss_terms = self.diffusion.training_losses(model=self.model, x_start=x_start, t=ts, noise=noise, model_kwargs=model_kwargs)

        self.log(f'train_loss', loss_terms['loss'].mean(), on_epoch=True, on_step=True, prog_bar=False)
        # for key in terms:
        #     losses.append(terms[key].float().mean())
        #     self.log(f'train_{key}', losses[key], on_epoch=True, on_step=True, prog_bar=False)

        return loss_terms['loss'].mean()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        real_imgs = batch["img"]
        cls = batch["cls"] if self.class_conditioned else None
        batch_size = real_imgs.shape[0]

        assert len(real_imgs.shape)==4 or (len(real_imgs.shape)==5 and batch_size==1), f"expected 4D (with any batch size) or 5D tensor(with batch size 1), got {real_imgs.shape}"

        imgs = torch.randn(batch_size,*(self.model_input_shape), device=self.device, dtype=real_imgs.dtype)

        model_kwargs = {"cls": cls}
        if self.task_type.startswith("superres"):
            low_res_img = batch["low_res_img"]
            model_kwargs["low_res"] = low_res_img

            imgs_to_log = []
            if len(low_res_img.shape)==4:
                low_res_imgs_tuple = low_res_img.cpu().split(1, dim=0)
            elif len(imgs.shape)==5:
                low_res_imgs_tuple = low_res_img[0].cpu().split(1, dim=-1)
            
            for i,img in enumerate(low_res_imgs_tuple):
                img = img.squeeze(0).squeeze(0).numpy()
                img = (img + 1) * 127.5
                img = img.clip(0, 255).astype(np.uint8)
                # imsave(f"/research/projects/m253231_Bardia/Current/04__Synthethic_Pelvis_Dataset/outputs/diffusion/imgs/{self.device}_{i}.png", img)
                imgs_to_log.append(img)
            self.logger.log_image(key="low-res samples", images=imgs_to_log)

        imgs = self.predict(imgs, model_kwargs=model_kwargs)

        imgs_to_log = []
        for i,img in enumerate(imgs):
            img = img.squeeze(0).squeeze(0).numpy()
            # imsave(f"/research/projects/m253231_Bardia/Current/04__Synthethic_Pelvis_Dataset/outputs/diffusion/superres/{self.device}_{i}.png", img)
            imgs_to_log.append(img)

        if self.class_conditioned:
            caption = []
            cls = cls.cpu().split(1, dim=0)
            for i,c in enumerate(cls):
                c = c.numpy().tolist()
                caption.append(f"Class: {c}")
            self.logger.log_image(key="validation samples", images=imgs_to_log, caption=caption)
        else:
            self.logger.log_image(key="validation samples", images=imgs_to_log)

    @torch.no_grad()
    def predict(self,imgs, model_kwargs=None, classifier_cond_scale=None, generator=None):            
        batch_size = imgs.shape[0]
        imgs = imgs.to(device=self.device, dtype=self.dtype)
        if model_kwargs is None:
            model_kwargs = {"cls": None}

        for key in model_kwargs:
            if model_kwargs[key] is not None:
                model_kwargs[key] = model_kwargs[key].to(device=self.device, dtype=self.dtype)
            
        indices = list(range(self.inference_diffusion.num_timesteps))[::-1]
        if self.inference_protocol == "DDPM":
            sample_fn = self.inference_diffusion.p_sample
            get_t = lambda i: self.timestep_map.gather(-1, torch.tensor([i]).long())
        elif self.inference_protocol.startswith("DDIM"):
            get_t = lambda i: torch.tensor([i]).long()
            sample_fn = self.inference_diffusion.ddim_sample
        else:
            raise ValueError(f"Unknown inference protocol {self.inference_protocol}, only DDPM, DDIM are supported")

        if classifier_cond_scale is None:
            classifier_cond_scale = self.classifier_cond_scale

        for i in tqdm(indices, desc="Creating Fake Images"):
            t = get_t(i)
            ts = t.expand(batch_size).to(device=self.device)
            out = sample_fn(
                self.model,
                imgs,
                ts,
                cond_scale=classifier_cond_scale,
                clip_denoised=True,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=model_kwargs,
                generator = generator,
                eta=0.0,
            )
            imgs = out["sample"].detach().to(dtype=imgs.dtype)
        
        if len(imgs.shape)==4: #2D
            imgs = imgs.split(1, dim=0)
        elif len(imgs.shape)==5: #3D
            imgs = imgs[0].split(1, dim=-1)

        imgs = list(imgs)

        for i in range(len(imgs)):
            img = imgs[i]
            img = (img + 1) * 127.5
            img = img.clip(0, 255).to(dtype=torch.uint8).cpu()
            imgs[i] = img

        return imgs

    def configure_optimizers(self):
        if self.total_steps is None:
            max_epochs=self.trainer.max_epochs
            grad_acc=self.trainer.accumulate_grad_batches
            self.set_total_steps(steps=len(self.train_dataloader())*max_epochs//grad_acc)

        params = bpu.add_weight_decay(self.model,5e-4)
        optimizer = torch.optim.AdamW(params, lr=self.lr) 
        return optimizer