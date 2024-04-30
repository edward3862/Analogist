import os
import torch
import argparse
import numpy as np
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from pipeline import InpaintingPipeline as Pipeline
from scheduling_ddim import DDIMScheduler as Scheduler
from attn_utils import AnalogistAttentionEdit as AttentionEdit
from attn_utils import regiter_attention_editor_diffusers
from utils import seed_everything, make_grid


class Analogist():
    def __init__(self, args):
        self.args = args
        seed_everything(args.seed)
        
        self.mask_image_fn = "mask_image_inpainting.png"
        self.mask_image = Image.open(self.mask_image_fn)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.generator = torch.Generator(device=self.device)
        self.generator = self.generator.manual_seed(args.seed)
        self.model_path = "runwayml/stable-diffusion-inpainting"
        self.model = Pipeline.from_pretrained(self.model_path, 
                                              torch_dtype=torch.float16, 
                                              safety_checker=None,).to(self.device)
        self.model.scheduler = Scheduler.from_config(self.model.scheduler.config)

        # Register attention editor
        self.attn_editor = AttentionEdit(sac_start_layer=args.sac_start_layer, sac_end_layer=args.sac_end_layer,
                                    cam_start_layer=args.cam_start_layer, cam_end_layer=args.cam_end_layer,
                                    scale_sac=args.scale_sac)
        regiter_attention_editor_diffusers(self.model, self.attn_editor)

        self.neg_prompt = "Messy,Disordered,Chaotic,Cluttered,Haphazard,Unkempt,Scattered,Disheveled,Tangled,Random"

    def prepare_output_dir(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        sample_count = len([item for item in os.listdir(output_dir) if item.startswith("sample_")])
        cur_output_dir = os.path.join(output_dir, f"sample_{sample_count}")
        os.makedirs(cur_output_dir, exist_ok=True)
        # os.system(f"cp {__file__} {cur_output_dir}")
        return cur_output_dir
    
    def run(self):
        args = self.args
        # Prepare output directory
        cur_output_dir = self.prepare_output_dir(args.output_dir)
        
        # Prepare grid image I
        image_fn = args.input_grid
        image = Image.open(image_fn)
        
        # Prepare paseted image I'
        height, width = args.res, args.res
        image = image.resize((width, height))
        self.mask_image = self.mask_image.resize((width, height))
        B = image.crop((0,height//2,width//2,height))
        image_pasted = Image.open(image_fn).resize((width,height))
        image_pasted.paste(B, (width//2,height//2))
        
        # Prepare noise x_T
        shape = (args.num_images_per_prompt, 4, height//8, width//8)
        latents = randn_tensor(shape, generator=None, dtype=torch.float16, device=self.device)

        # Get GPT4 prompts
        with open(args.prompt_file, "r") as f:
            prompts = f.read().strip().split("\n")
        a_prompt="best quality, extremely detailed"
        if a_prompt:
            prompts = [p + f", {a_prompt}" for p in prompts]

        # Run the model
        for prompt in prompts:
            postfix = prompt.replace(" ", "_").replace(",", "").replace(".", "")
            output = self.model(prompt,
                           image=image_pasted,
                           mask_image=self.mask_image,
                           latents=latents,
                           guidance_scale=args.guidance_scale,
                           negative_prompt=self.neg_prompt,
                           num_images_per_prompt=args.num_images_per_prompt,
                           generator=self.generator,
                           strength=args.strength,
                           height=height,
                           width=width
                        ).images
            out_image = [np.array(image), np.array(image_pasted)] \
                + [np.array(img.crop((width//2,height//2,width,height)).resize((width,height))) for img in output]
            out_image = np.concatenate(out_image, axis=1)
            out_image = Image.fromarray(out_image)
            out_path = os.path.join(cur_output_dir, f"{postfix}.png")
            out_image.save(out_path)
            print("Syntheiszed images are saved in", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analogist')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_grid', type=str, default="data/colorization_processed/input.png")
    parser.add_argument('--prompt_file', type=str, default="data/colorization_processed/prompts.txt")
    parser.add_argument('--output_dir', type=str, default="output/colorization")
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--num_images_per_prompt', type=int, default=8)
    parser.add_argument('--strength', type=float, default=0.96)
    parser.add_argument('--guidance_scale', type=float, default=15)
    parser.add_argument('--sac_start_layer', type=int, default=3)
    parser.add_argument('--sac_end_layer', type=int, default=11)
    parser.add_argument('--cam_start_layer', type=int, default=3)
    parser.add_argument('--cam_end_layer', type=int, default=11)
    parser.add_argument('--scale_sac', type=float, default=1.3)
    args = parser.parse_args()
    
    analogist = Analogist(args)
    analogist.run()