!pip install diffusers transformers accelerate imageio imageio-ffmpeg --quiet

from diffusers import StableVideoDiffusionPipeline
import torch
from PIL import Image
import imageio

# Load model
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid", 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")

# Upload gambar
from google.colab import files
uploaded = files.upload()

image_path = list(uploaded.keys())[0]
input_image = Image.open(image_path)

# Generate video
video_frames = pipe(input_image, num_frames=14).frames[0]

# Simpan video
output_path = "hasil_video.mp4"
imageio.mimsave(output_path, video_frames, fps=8)

print("Video berhasil dibuat →", output_path)
