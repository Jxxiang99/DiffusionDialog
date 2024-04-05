import argparse
import inspect

from . import gaussian_diffusion as gd
from model.bart_diffusion.diffusion.respace import SpacedDiffusion, space_timesteps


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    return dict(
        learn_sigma=False,
        sigma_small=False,
        diffusion_steps=2000,
        noise_schedule="sqrt",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=True,
    )

def create_gaussian_diffusion(
    *,
    diffusion_steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=True,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    #确定加噪方案
    #loss_type:RESCALED_MSE
    #mean_type:EPSILON
    #var_type:FIXED_LARGE
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
