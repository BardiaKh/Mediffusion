def extract_diffusion_args(diffusion):
    betas = diffusion.betas
    model_mean_type = diffusion.model_mean_type
    model_var_type = diffusion.model_var_type
    loss_type = diffusion.loss_type
    rescale_timesteps = diffusion.rescale_timesteps
    p2_gamma =diffusion.p2_gamma
    p2_k = diffusion.p2_k
    verbose = diffusion.verbose

    kwargs = {
        "betas": betas,
        "model_mean_type": model_mean_type,
        "model_var_type": model_var_type,
        "loss_type": loss_type,
        "rescale_timesteps": rescale_timesteps,
        "p2_gamma": p2_gamma,
        "p2_k": p2_k,
        "verbose": verbose,
    }

    return kwargs
