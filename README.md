# modified-euler-samplers-for-sonar-diffusers

----

This repo ports the **momentum mechanism** of [sd-webui-sonar](https://github.com/Kahsolt/stable-diffusion-webui-sonar) on `Euler` and `Euler a` sampler to [huggingface/diffusers](https://github.com/huggingface/diffusers). 

how to use assuming you initiated a pipe in diffusers:
```
pipe.scheduler = EulerA.from_config(pipe.scheduler.config)
```
