# modified-euler-samplers-for-sonar-diffusers

----

This repo ports the **momentum mechanism** of [sd-webui-sonar](https://github.com/Kahsolt/stable-diffusion-webui-sonar) on `Euler` and `Euler a` sampler to [huggingface/diffusers](https://github.com/huggingface/diffusers). 

how to use assuming you initiated a pipe in diffusers:
```
pipe.scheduler = EulerA.from_config(pipe.scheduler.config)
```
# change sonar settings
after initiating the scheduler, write these lines (whichever ones you'd like to modify):
```
pipe.scheduler.history_d=0 #'rand_new' or 'rand_init' are all possible values for this
pipe.scheduler.momentum=0.90 #
pipe.scheduler.momentum_hist=0.9
```
here are the default values:
```
history_d=0
momentum=0.95
momentum_hist=0.75
```
