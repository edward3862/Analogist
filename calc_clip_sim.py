import os
import torch
import numpy as np
from PIL import Image
from utils import decompose_grid
from transformers import CLIPProcessor, CLIPModel
from torch.nn.functional import cosine_similarity

def run_clip(model, processor, text, image, device="cuda"):
    inputs = processor(text=None, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(inputs.pixel_values).to(device)
    
    image_feature = image_features[0].view(1, -1)
    return None, image_feature


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "openai/clip-vit-base-patch16"  # Choose the appropriate CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    datasets = {
        "low_level_tasks": ["colorization", "deblur", "denoise", "enhancement"],
        "manipulation_tasks": ["image_editing", "image_translation", "style_transfer"],
        "vision_tasks": ["ubcfashion", "scannet", "inpainting"]
    }
    avg_results = []
    for category in datasets.keys():
        tasks = datasets[category]
        for task in tasks:
            my_result_root = os.path.join("results", category, task)
            
            result = []
            fns = sorted(os.listdir(my_result_root))

            for file in fns:
                fn = os.path.join(my_result_root, file)
                image = Image.open(fn)

                input_image = image.crop((0, 0, 512, 512))
                img_A, img_Aprime, img_B, _ = decompose_grid(input_image)
                _, img_feat_A = run_clip(model, processor, None, [img_A], device="cuda")
                _, img_feat_Aprime = run_clip(model, processor, None, [img_Aprime], device="cuda")
                _, img_feat_B = run_clip(model, processor, None, [img_B], device="cuda")
                
                similarities = []
                for j in range(2,10,1):
                    img_Bprime = image.crop((512*j, 0, 512*j+512, 512)).resize((256, 256))
                    _, img_feat_Bprime = run_clip(model, processor, None, [img_Bprime], device="cuda")
                    
                    img_direction_A = img_feat_Aprime - img_feat_A
                    img_direction_B = img_feat_Bprime - img_feat_B

                    similarity = cosine_similarity(img_direction_A, img_direction_B)
                    similarities.append(similarity.item())
                result.append(np.mean(similarities))

            print(f"{task}:\t{np.mean(result)}")
            avg_results.append(np.mean(result))

    print(f"Average similarity: {np.mean(avg_results)}")