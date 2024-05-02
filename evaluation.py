import os
import torch
import argparse
import numpy as np
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from analogist import Analogist


def load_prompt(prompt_file, prompt_index=0, a_prompt=None):
    with open(prompt_file, "r") as f:
        prompts = f.readlines()
    prompts = [p.strip().split('\t')[1:][prompt_index] for p in prompts]
    if a_prompt is not None:
        prompts = [p + ', ' + a_prompt for p in prompts]
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analogist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_dir', type=str, default='datasets/low_level_tasks_processed/colorization_masked')
    parser.add_argument('--prompt_file', type=str, default="datasets/low_level_tasks_processed/colorization_gpt4_out.txt")
    parser.add_argument('--output_dir', type=str, default="results/dev")
    parser.add_argument('--num_samples', type=int, default=-1)
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--num_images_per_prompt', type=int, default=8)
    parser.add_argument('--strength', type=float, default=0.98)
    parser.add_argument('--guidance_scale', type=float, default=15)
    parser.add_argument('--sac_start_layer', type=int, default=3)
    parser.add_argument('--sac_end_layer', type=int, default=11)
    parser.add_argument('--cam_start_layer', type=int, default=3)
    parser.add_argument('--cam_end_layer', type=int, default=11)
    parser.add_argument('--scale_sac', type=float, default=1.3)
    args = parser.parse_args()
    
    analogist = Analogist(args)

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Prepare mask image
    height, width = args.res, args.res
    mask_image = analogist.mask_image.resize((width, height))

    # Prepare the dataset
    fns = sorted(os.listdir(input_dir))
    prompts = load_prompt(args.prompt_file, prompt_index=0, 
                               a_prompt="best quality, extremely detailed")
    assert len(fns) == len(prompts)
    num_samples = args.num_samples
    if num_samples == -1:
        num_samples = len(prompts)

    # Run Analogist
    for fn, gpt4_prompt in zip(fns[:num_samples], prompts[:num_samples]):
        
        # Load image I
        in_fn = os.path.join(input_dir, fn)
        image = Image.open(in_fn)
        
        # Prepare paseted image I'
        B = image.crop((0,height//2,width//2,height))
        image_pasted = Image.open(in_fn)
        image_pasted.paste(B, (width//2,height//2))

        # Prepare noise
        shape = (args.num_images_per_prompt, 4, height//8, width//8)
        latents = randn_tensor(shape, generator=analogist.generator, dtype=torch.float16, device=analogist.device)

        # Run the model
        output = analogist.inpaint(gpt4_prompt, image_pasted, latents, height, width)

        # Save the output
        out_image = [np.array(image), np.array(image_pasted)] \
            + [np.array(img.crop((width//2,height//2,width,height)).resize((width,height))) for img in output]
        out_image = np.concatenate(out_image, axis=1)
        out_image = Image.fromarray(out_image)
        out_path = os.path.join(output_dir, fn)
        out_image.save(out_path)
        print("Syntheiszed images are saved in", out_path)
