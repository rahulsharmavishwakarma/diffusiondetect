import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import gradio as gr
from PIL import Image

model_id = "stabilityai/stable-diffusion-2-1"

def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    generated_image = pipe(prompt).images[0]

    # Use YOLOv5 for object detection
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    results = model(generated_image)
    filtered_image = results.render()[0]

    return generated_image, filtered_image

# Create Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter your image prompt..."),
    outputs=[
        gr.Image(type="numpy", label="Generated Image"),
        gr.Image(type="numpy", label="Filtered Image"),
    ],
    title="DiffusionDetect",
    description="Generate an image based on a text prompt and apply object detection.",
)

# Launch the interface
iface.launch(share=True)
