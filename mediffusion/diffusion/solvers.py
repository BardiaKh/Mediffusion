import os
import numpy as np
import torch
import PIL
from PIL import Image
from tqdm.auto import tqdm
from .base import GaussianDiffusionBase
from ..utils.diffusion import get_respaced_betas
from ..utils.solvers import extract_diffusion_args

class SolverBase(GaussianDiffusionBase):
    def __init__(self, **diffusion_params):
        super().__init__(**diffusion_params)

    def _get_t(self, i):
        raise NotImplementedError

    def _get_model_device(self, model):
        return next(model.parameters()).device
    
    def _get_model_dtype(self, model):
        return next(model.parameters()).dtype

    def _sample_fn(*args,**kwargs):
        raise NotImplementedError

    def sample(*args,**kwargs):
        raise NotImplementedError    
    
class DDPMSolver(SolverBase):
    def __init__(self, diffusion):
        kwargs = extract_diffusion_args(diffusion)
        super().__init__(**kwargs)

    def _get_t(self, i):
        return torch.tensor([i]).long()

    def _sample_fn(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        generator=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond_scale=cond_scale,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn(*x.shape, device=x.device, dtype=x.dtype, generator=generator)
        nonzero_mask = (
            (t != 0).to(x.dtype).view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    @torch.no_grad()
    def sample(self, model, imgs, start_denoise_step=None, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None, mask=None, original_image=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(start_denoise_step))[::-1]

        for i in tqdm(indices, desc="DDPM Sampling", disable = not self.verbose):
            t = self._get_t(i)
            ts = t.expand(batch_size).to(device=device)

            if mask is not None:
                x0_noised = self.q_sample(original_image, ts)
                imgs = imgs * mask + x0_noised * (1 - mask)
                imgs = imgs.to(device=device, dtype=dtype)

            out = self._sample_fn(
                model,
                imgs,
                ts,
                cond_scale=cond_scale,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                generator = generator,
            )
            imgs = out["sample"].detach().to(dtype=dtype, device=device)

        if mask is not None:
            imgs = imgs * mask + original_image * (1 - mask)
            imgs = imgs.to(device=device, dtype=dtype)

        return imgs
    
class DDPMDumpSolver(SolverBase):
    def __init__(self, diffusion):
        kwargs = extract_diffusion_args(diffusion)
        super().__init__(**kwargs)

    def _get_t(self, i):
        return torch.tensor([i]).long()

    def _sample_fn(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        generator=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param os.environ["DDPM_DUMP_OUTPUT_BASENAME"]: the base filename to write to.
            images will be written with the timestep appended.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond_scale=cond_scale,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn(*x.shape, device=x.device, dtype=x.dtype, generator=generator)
        nonzero_mask = (
            (t != 0).to(x.dtype).view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
    
    @torch.no_grad()
    def sample(self, model, imgs, start_denoise_step=None, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(start_denoise_step))[::-1]

        for i in tqdm(indices, desc="DDPM Sampling and dumping", disable = not self.verbose):
            t = self._get_t(i)
            ts = t.expand(batch_size).to(device=device)
            out = self._sample_fn(
                model,
                imgs,
                ts,
                cond_scale=cond_scale,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                generator = generator,
            )
            imgs = out["sample"].detach().to(dtype=dtype, device=device)
              
            # Dump 0th image to file
            jj = 0  # index to save  
            # Wow, how many ways can there be to cast a pile of numbers???  Nine and counting!
            imgjj1 = imgs[jj].squeeze().to('cpu')      # convert to something (type "<f2")?
            imgjj2 = imgjj1.float()                    # convert to full precision floats (i think)
            imgjj3 = (255 * (imgjj2 + 1) / 2)          # [-1,1] -> [0,255]
            imgjj4 = imgjj3.numpy().astype(np.uint8)   # convert to ints
            img = PIL.Image.fromarray(imgjj4).convert('L')  # greyscale PIL image
            out_filename = os.environ["DDPM_DUMP_OUTPUT_BASENAME"] + f"-{jj:02d}.{ts[jj]:03d}.png"
            img.save(out_filename, "png")

        return imgs
    
class DDIMSolver(SolverBase):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, diffusion, num_steps, eta=0.0):
        kwargs = extract_diffusion_args(diffusion)
        base_diffusion = GaussianDiffusionBase(**kwargs)

        _, self.use_timesteps = get_respaced_betas(base_diffusion.betas, f"DDIM{num_steps}")
                    
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
                
        kwargs["betas"] = torch.tensor(new_betas)
        self.eta = eta
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
    
    def _get_t(self, i):
        return torch.tensor([i]).long()
    
    def _sample_fn(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        generator=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond_scale=cond_scale,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = self._extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = self._extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta * \
            torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * \
            torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn(*x.shape, device=x.device, dtype=x.dtype, generator=generator)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + \
            torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}
        
    @torch.no_grad()
    def sample(self, model, imgs, start_denoise_step=None, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None, mask=None, original_image=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(start_denoise_step))[::-1]

        for i in tqdm(indices, desc="DDIM Sampling", disable = not self.verbose):
            t = self._get_t(i)            
            ts = t.expand(batch_size).to(device=device)

            if mask is not None:
                x0_noised = self.q_sample(original_image, ts)
                imgs = imgs * mask + x0_noised * (1 - mask)
                imgs = imgs.to(device=device, dtype=dtype)

            out = self._sample_fn(
                model,
                imgs,
                ts,
                cond_scale=cond_scale,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                generator=generator,
                eta=self.eta,
            )
            imgs = out["sample"].detach().to(dtype=dtype, device=device)

        if mask is not None:
            imgs = imgs * mask + original_image * (1 - mask)
            imgs = imgs.to(device=device, dtype=dtype)

        return imgs
        
class InverseDDIMSolver(SolverBase):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, diffusion, num_steps, eta=0.0):
        kwargs = extract_diffusion_args(diffusion)
        base_diffusion = GaussianDiffusionBase(**kwargs)

        _, self.use_timesteps = get_respaced_betas(base_diffusion.betas, f"DDIM{num_steps}")
                    
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
                
        kwargs["betas"] = torch.tensor(new_betas)
        self.eta = 0.0
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
    
    def _get_t(self, i):
        return torch.tensor([i]).long()
        
    def _sample_fn(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond_scale=cond_scale,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * \
            x - out["pred_xstart"]
        ) / self._extract_into_tensor(self.sqrt_recip_alphas_cumprod_minus_one, t, x.shape)
        alpha_bar_next = self._extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next) * eps

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}
    
    @torch.no_grad()
    def sample(self, model, imgs, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None, start_denoise_step=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        indices = list(range(self.num_timesteps))
        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))
        else:
            indices = list(range(start_denoise_step, self.num_timesteps))

        for i in tqdm(indices, desc="Creating DDIM Noise", disable = not self.verbose):
            t = self._get_t(i)
            ts = t.expand(batch_size).to(device=device)
            out = self._sample_fn(
                model,
                imgs,
                ts,
                cond_scale=cond_scale,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                eta=self.eta,
            )
            imgs = out["sample"].detach().to(dtype=dtype, device=device)

        return imgs

class PLMSSolver(SolverBase):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, diffusion, num_steps):
        kwargs = extract_diffusion_args(diffusion)
        base_diffusion = GaussianDiffusionBase(**kwargs)

        _, self.use_timesteps = get_respaced_betas(base_diffusion.betas, f"DDIM{num_steps}")
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
                
        kwargs["betas"] = torch.tensor(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
    
    def _get_t(self, i):
        return torch.tensor([i]).long()
    
    def _get_eps(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            cond_scale=cond_scale,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        return eps
    
    def _eps_to_pred_xstart(
        self,
        x,
        eps,
        t,
    ):
        alpha_bar = self._extract_into_tensor_lerp(self.alphas_cumprod, t, x.shape)
        return (x - eps * torch.sqrt(1 - alpha_bar)) / torch.sqrt(alpha_bar)

    def _pndm_transfer(
        self,
        x,
        eps,
        t_1,
        t_2,
    ):
        pred_xstart = self._eps_to_pred_xstart(x, eps, t_1)
        alpha_bar_prev = self._extract_into_tensor_lerp(self.alphas_cumprod, t_2, x.shape)
        return (pred_xstart * torch.sqrt(alpha_bar_prev) + torch.sqrt(1 - alpha_bar_prev) * eps).to(x.dtype)

    def _prk_sample_fn(
        self,
        model,
        x,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using fourth-order Pseudo Runge-Kutta
        (https://openreview.net/forum?id=PlKWVd2yBkY).
        Same usage as p_sample().
        """
        if model_kwargs is None:
            model_kwargs = {}

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        t_mid = t - 1
        t_prev = t - 2
        eps_1 = self._get_eps(model, x, t, cond_scale=cond_scale, model_kwargs=model_kwargs, cond_fn=cond_fn)
        x_1 = self._pndm_transfer(x, eps_1, t, t_mid)
        eps_2 = self._get_eps(model, x_1, t_mid, cond_scale=cond_scale, model_kwargs=model_kwargs, cond_fn=cond_fn)
        x_2 = self._pndm_transfer(x, eps_2, t, t_mid)
        eps_3 = self._get_eps(model, x_2, t_mid, cond_scale=cond_scale, model_kwargs=model_kwargs, cond_fn=cond_fn)
        x_3 = self._pndm_transfer(x, eps_3, t, t_prev)
        eps_4 = self._get_eps(model, x_3, t_prev, cond_scale=cond_scale, model_kwargs=model_kwargs, cond_fn=cond_fn)
        eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6

        sample = self._pndm_transfer(x, eps_prime, t, t_prev)
        pred_xstart = self._eps_to_pred_xstart(x, eps_prime, t)
        pred_xstart = process_xstart(pred_xstart)
        return {"sample": sample, "pred_xstart": pred_xstart, "eps": eps_prime}
    
    def _sample_prk(self, model, imgs, start_denoise_step=None, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(start_denoise_step))[::-1]

        indices = indices[1:-1]

        for i in tqdm(indices, desc="PRK Sampling", disable = not self.verbose):
            t = self._get_t(i)            
            ts = t.expand(batch_size).to(device=device)
            out = self._prk_sample_fn(
                model,
                imgs,
                ts,
                cond_scale=cond_scale,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                generator=generator,
            )
            imgs = out["sample"].detach().to(dtype=dtype, device=device)

        return imgs

    def _plms_sample_fn(
        self,
        model,
        x,
        old_eps,
        t,
        cond_scale=0,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using fourth-order Pseudo Linear Multistep
        (https://openreview.net/forum?id=PlKWVd2yBkY).
        """
        if model_kwargs is None:
            model_kwargs = {}

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        eps = self._get_eps(model, x, t, cond_scale=cond_scale, model_kwargs=model_kwargs, cond_fn=cond_fn)
        eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        sample = self._pndm_transfer(x, eps_prime, t, t - 1).to(x.dtype)
        pred_xstart = self._eps_to_pred_xstart(x, eps, t)
        pred_xstart = process_xstart(pred_xstart)
        return {"sample": sample, "pred_xstart": pred_xstart, "eps": eps}
    
    @torch.no_grad()
    def sample(self, model, imgs, start_denoise_step=None, cond_scale=1, clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None, generator=None, mask=None, original_image=None):
        device = self._get_model_device(model)
        dtype = self._get_model_dtype(model)
        
        imgs = imgs.to(device=device, dtype=dtype)
        batch_size = imgs.shape[0]

        if start_denoise_step is None:
            indices = list(range(self.num_timesteps))[::-1]
        else:
            indices = list(range(start_denoise_step))[::-1]

        indices = indices[1:-1]

        old_eps = []

        for i in tqdm(indices, desc="PLMS Sampling", disable = not self.verbose):
            t = self._get_t(i)            
            ts = t.expand(batch_size).to(device=device)

            if mask is not None:
                x0_noised = self.q_sample(original_image, ts)
                imgs = imgs * mask + x0_noised * (1 - mask)
                imgs = imgs.to(device=device, dtype=dtype)

            if len(old_eps) < 3:
                out = self._prk_sample_fn(
                    model,
                    imgs,
                    ts,
                    cond_scale=cond_scale,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            else:
                out = self._plms_sample_fn(
                    model,
                    imgs,
                    old_eps,
                    ts,
                    cond_scale=cond_scale,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                old_eps.pop(0)
            old_eps.append(out["eps"])
            imgs = out["sample"].detach().to(dtype=dtype, device=device)

        if mask is not None:
            imgs = imgs * mask + original_image * (1 - mask)
            imgs = imgs.to(device=device, dtype=dtype)

        return imgs

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)

    def forward_with_cond_scale(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        ts = ts.long()
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model.forward_with_cond_scale(x, new_ts, **kwargs)
