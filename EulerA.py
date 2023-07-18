from diffusers import EulerAncestralDiscreteScheduler
from torch import Tensor
import torch
from typing import Callable, List, Optional, Tuple, Union, Dict, Any, Literal
from diffusers.utils import randn_tensor
from diffusers.configuration_utils import ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
class EulerA(EulerAncestralDiscreteScheduler, SchedulerMixin, ConfigMixin):
    history_d=0
    momentum=0.95
    momentum_hist=0.75
    def init_hist_d(self,x:Tensor) -> Union[Literal[0], Tensor]:
        # memorize delta momentum
        if   self.history_d == 0:      self.history_d = 0
        elif self.history_d == 'rand_init': self.history_d = x
        elif self.history_d == 'rand_new':  self.history_d = torch.randn_like(x)
        else: raise ValueError(f'unknown momentum_hist_init: {self.history_d}')
    def momentum_step(self, x:Tensor, d:Tensor, dt:Tensor):
        hd=self.history_d
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
        self.history_d=hd
        return x
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
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
            generator (`torch.Generator`, optional): Random number generator.
            return_dict (`bool`): option for returning tuple rather than EulerAncestralDiscreteSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.EulerAncestralDiscreteSchedulerOutput`] if `return_dict` is True, otherwise
            a `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if not isinstance(self.history_d, torch.Tensor) and not isinstance(self.history_d, int):
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

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)

        step_index = (self.timesteps == timestep).nonzero().item()
        sigma = self.sigmas[step_index]

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            pred_original_sample = sample - sigma * model_output
        elif self.config.prediction_type == "v_prediction":
            # * c_out + input * c_skip
            pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        sigma_from = self.sigmas[step_index]
        sigma_to = self.sigmas[step_index + 1]
        sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
        sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

        # 2. Convert to an ODE derivative
        derivative = (sample - pred_original_sample) / sigma

        dt = sigma_down - sigma
        
        prev_sample = self.momentum_step(sample,derivative,dt)

        device = model_output.device
        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

        prev_sample = prev_sample + noise * sigma_up

        if not return_dict:
            return (prev_sample,)
        output={prev_sample:prev_sample, pred_original_sample:pred_original_sample}
        return output