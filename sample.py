from diffusers.pipelines import StableDiffusionPipeline
import torch
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--prompt", type=str, default="a photo of dog")
parser.add_argument("--save_dir", type=str, default="data/dogs/class")
parser.add_argument("--sample_nums", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=16)
args = parser.parse_args()

sample_nums = args.sample_nums
batch_size = args.batch_size
prompt = args.prompt
save_dir = args.save_dir


if __name__ == "__main__":
    os.makedirs(save_dir, exist_ok=True)
    
    model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    datasets = [prompt] * sample_nums
    datasets = [datasets[x:x+batch_size] for x in range(0, sample_nums, batch_size)]
    id = 0

    for text in datasets:
        with torch.no_grad():
            images = model(text, height=512, width=512, num_inference_steps=50).images

        for image in images:
            image.save(f"{save_dir}/{id}.png")
            id += 1
