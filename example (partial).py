from schedulers.EulerA import EulerA
# Initialize the Celery app

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16,local_files_only=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
  "runwayml/stable-diffusion-v1-5",controlnet=controlnet,local_files_only=True, torch_dtype=torch.float16,safety_checker=None, requires_safety_checker=False,
).to('cuda')
pipe.scheduler=EulerA.from_config(pipe.scheduler.config)  # import one of the 2 schedulers from this repo
pipe.scheduler.history_d='rand_new' #0 'rand_new' or 'rand_init'
pipe.scheduler.momentum=0.95 #number should be between -1 and 1
pipe.scheduler.momentum_hist=0.75  #number should be between -1 and 1
buffer=open('img0.png', 'rb')
buffer.seek(0)
image_bytes = buffer.read()
images = Image.open(BytesIO(image_bytes))
start_time = time.time()
generator = torch.manual_seed(2733424006)

image=pipe(
    "A person standing in a field of flowers, 4k, realistic",
    images,
    num_inference_steps=20,
    height=512,
    width=512,
    generator=generator
).images[0]
end_time = time.time()
execution_time = end_time - start_time
print("Execution time: {:.2f} seconds".format(execution_time))
# print(image)
image.save('img1.png', format='PNG')