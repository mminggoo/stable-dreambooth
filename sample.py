from diffusers.pipelines import StableDiffusionPipeline
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
parser.add_argument("--prompt", type=str, default="a photo of dog")
parser.add_argument("--save_dir", type=str, default="data/dogs/class")
args = parser.parse_args()

sample_nums = 1000
batch_size = 16
prompt = args.prompt
save_dir = args.save_dir


if __name__ == "__main__":
    model_id = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    datasets = [prompt] * sample_nums
    datasets = [datasets[x:x+batch_size] for x in range(0, sample_nums, batch_size)]
    id = 0

    for text in datasets:
        with torch.no_grad():
            images = model(text, height=512, width=512, num_inference_steps=50)["sample"]

        for image in images:
            image.save(f"{save_dir}/{id}.png")
            id += 1
