from .base_pipeline import BasePipeline
from ..core.uneven_tensor import UnevenTensor
import torch
import torch.nn.functional as F


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(
        inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps)
    )
    min_timestep_boundary = int(
        inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps)
    )

    input_latents = inputs["input_latents"]


    if isinstance(input_latents, UnevenTensor):

        batch_size = input_latents.batch_size
        timestep_ids = torch.randint(
            min_timestep_boundary, max_timestep_boundary, (batch_size,)
        )
        timesteps = pipe.scheduler.timesteps[timestep_ids].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )


        noise = UnevenTensor.apply(torch.randn_like, input_latents)


        inputs["latents"] = UnevenTensor.apply(
            lambda x, n, t: pipe.scheduler.add_noise(x, n, t),
            input_latents,
            noise,
            timesteps.tolist(),
        )


        training_target = UnevenTensor.apply(
            lambda x, n, t: pipe.scheduler.training_target(x, n, t),
            input_latents,
            noise,
            timesteps.tolist(),
        )

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

        noise_pred = pipe.model_fn(**models, **inputs, timestep=timesteps)


        loss = 0
        for np, tt, t in zip(noise_pred, training_target, timesteps):
            weight = pipe.scheduler.training_weight(t)
            loss = loss + torch.nn.functional.mse_loss(np.float(), tt.float()) * weight
        loss = loss / batch_size
    else:

        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype, device=pipe.device
        )

        noise = torch.randn_like(input_latents)
        inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
        training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * pipe.scheduler.training_weight(timestep)

    return loss
