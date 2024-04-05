from sched import scheduler
from model.bart_diffusion.diffusion.script_util import create_gaussian_diffusion,model_and_diffusion_defaults
from model.bart_diffusion.diffusion.resample import ScheduleSampler, create_named_schedule_sampler
import torch
def get_gaussian_diffusion_and_sampler(timestep_respacing=""):
    dict = model_and_diffusion_defaults()
    dict['timestep_respacing'] = timestep_respacing
    diffusion = create_gaussian_diffusion(**dict)
    #返回训练时的diffusion t采样器
    schedule_sampler = create_named_schedule_sampler('uniform', diffusion)
    # if checkpoint is None:
    #     model = LatentTrasformer(config, diffusion.num_timesteps)
    # elif config is None:
    #     path = checkpoint + '/latent_transformer.pth'
    #     model = torch.load(path)
        
    # return diffusion, model, schedule_sampler
    return diffusion, schedule_sampler