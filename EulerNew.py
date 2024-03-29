from diffusers import EulerDiscreteScheduler
from torch import Tensor
import torch
from typing import Callable, List, Optional, Tuple, Union, Dict, Any, Literal
from diffusers.utils import BaseOutput
try:
    # Try the old import path
    from diffusers.utils import randn_tensor
except ImportError:
    # If the old import path is not available, use the new import path
    from diffusers.utils.torch_utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
class Output(BaseOutput):
    """
    Output class for the scheduler's step function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None
class Euler(EulerDiscreteScheduler, SchedulerMixin, ConfigMixin):
    history_d=0
    momentum=0.95
    momentum_hist=0.75
    used_history_d=None
    def init_hist_d(self,x:Tensor) -> Union[Literal[0], Tensor]:
        # memorize delta momentum
        if   self.history_d == 0:      self.used_history_d = 0
        elif self.history_d == 'rand_init': self.used_history_d = x
        elif self.history_d == 'rand_new':  self.used_history_d = torch.randn_like(x)
        else: raise ValueError(f'unknown momentum_hist_init: {self.history_d}')
    # def add_noise(
    #     self,
    #     original_samples: torch.FloatTensor,
    #     noise: torch.FloatTensor,
    #     timesteps: torch.FloatTensor,
    # ) -> torch.FloatTensor:
    #     # Make sure sigmas and timesteps have the same device and dtype as original_samples
    #     sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    #     if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
    #         # mps does not support float64
    #         schedule_timesteps = self.timesteps.to(original_samples.device, dtype=torch.float32)
    #         timesteps = timesteps.to(original_samples.device, dtype=torch.float32)
    #     else:
    #         schedule_timesteps = self.timesteps.to(original_samples.device)
    #         timesteps = timesteps.to(original_samples.device)
    #     step_indices = []
    #     print(212121221,timesteps)
    #     for t in timesteps:
    #         # Find the indices where schedule_timesteps is equal to t
    #         indices = (schedule_timesteps == t).nonzero()
    #         print(6666,indices)
    #         print(4444,schedule_timesteps,t)
    #         # Check if any indices were found
    #         if indices.numel() > 0:
    #             # Extract the first index as a scalar (assuming you want the first match)
    #             index = indices[0].item()
    #             step_indices.append(index)
    #         else:
    #             # Handle the case where no matching index was found
    #             step_indices.append(None)
    #     print(29292,step_indices)
    #     sigma = sigmas[step_indices].flatten()
    #     while len(sigma.shape) < len(original_samples.shape):
    #         sigma = sigma.unsqueeze(-1)

    #     noisy_samples = original_samples + noise * sigma
    #     return noisy_samples
    def momentum_step(self, x:Tensor, d:Tensor, dt:Tensor):
        hd=self.used_history_d
        # correct current `d` with momentum
        p = 1.0 - self.momentum
        self.momentum_d = (1.0 - p) * d + p * hd
        # Euler method with momentum
        x = x + self.momentum_d * dt

        # update momentum history
        q = 1.0 - self.momentum_hist
        if (isinstance(hd, int) and hd == 0):
            hd = self.momentum_d
        else:
            hd = (1.0 - q) * hd + q * self.momentum_d
        self.used_history_d=hd
        return x
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ):
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`float`): current timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            s_churn (`float`)
            s_tmin  (`float`)
            s_tmax  (`float`)
            s_noise (`float`)
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.EulerDiscreteSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if not isinstance(self.used_history_d, torch.Tensor) and not isinstance(self.used_history_d, int):
            self.init_hist_d(sample)
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if not self.is_scale_input_called:
            logger.warning(
                "The `scale_model_input` function should be called before `step` to ensure correct denoising. "
                "See `StableDiffusionPipeline` for a usage example."
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        if self.step_index is None:
            self._init_step_index(timestep)
        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(
            model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator
        )

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility
        if self.config.prediction_type == "original_sample" or self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma_hat * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = self.momentum_step(sample,derivative,dt)
        # print(111111,pred_original_sample.shape)
        self._step_index+=1
        if self._step_index==(len(self.sigmas)-1):
            self.used_history_d=None
        if not return_dict:
            return (prev_sample,)
    
        return Output(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample
        )