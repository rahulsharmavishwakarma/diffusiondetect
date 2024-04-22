import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "Showcase the cup in a cozy home setting, with a person enjoying a hot beverage while reading a book or working on their laptop."
image = pipe(prompt).images[0]
gen_img = image  
image.save("genimage.png")

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

# Images
img = gen_img # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.save("./filterimage")  # or .show(), .save(), .crop(), .pandas(), etc.

