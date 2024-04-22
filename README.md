# DiffusionDetect

DiffusionDetect is a project that combines the power of Stable Diffusion models for image generation with object detection using YOLOv5. This project allows users to generate images based on text prompts using the Stable Diffusion model and then apply object detection to the generated images using YOLOv5.

<img src="https://github.com/rahulsharmavishwakarma/diffusiondetect/blob/main/examples/imagef.webp" alt="Image description" width="1000" height="">

## Features

- Generate images based on text prompts using Stable Diffusion models.
- Apply object detection to the generated images using YOLOv5.
- View both the generated and filtered images side by side in a user-friendly interface.

## Getting Started

To run the project, follow these steps:

### 1. Install Required Libraries

Install the required libraries by running the following command in your terminal:

```bash
pip install torch diffusers gradio
```
### 2. Clone the project repository by running the following command:

```bash
git clone github.com/rahulsharmavishwakarma/diffusiondetect.git
```
### 3. Run the provided Python script to launch the Gradio interface for image generation and object detection. Navigate to the project directory and run the following command:

```bash
python app.py
```
## Usage
- Enter a text prompt in the input textbox on the Gradio interface.
- Click the "Generate" button to create an image based on the prompt.
- View the generated image and the filtered image with object detection results side by side.
- Explore different prompts and observe the variations in generated images and object detection outcomes.
## Dependencies
-  torch
-  diffusers
-  gradio
-  PIL
-  ultralytics/yolov5
## Acknowledgements
- Stable Diffusion models by StabilityAI
- YOLOv5 model by Ultralytics
