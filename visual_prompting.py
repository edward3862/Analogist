import os
import cv2
import argparse
import numpy as np

from PIL import Image
from PIL import ImageDraw, ImageFont


def add_mark(image, text, coord, size, font_color=(255, 255, 255)):
    x, y = coord
    width=size
    height=size
    fontsize=size
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", fontsize)
    draw.rectangle([x, y, x + width, y + height], fill="black")
    draw.text((x+2, y), text, font=font, fill=font_color)
    return image


def add_marks_to_grid(image, grid_size, p, size):
    image = add_mark(image, "A", (0+p, 0+p), size)
    image = add_mark(image, "A'", (grid_size+p-1, 0+p), size)
    image = add_mark(image, "B", (0+p, grid_size+p-1), size)
    image = add_mark(image, "B'", (grid_size+p-1, grid_size+p-1), size)
    return image


def add_arrows_to_grid(im, grid_size=256, hl=25, size=5):
    hl = int(grid_size / 10)
    size = int(grid_size / 64)
    na = np.array(im)
    na = cv2.arrowedLine(na, (grid_size-hl,grid_size//2), 
                         (grid_size+hl, grid_size//2), (230,0,0), 
                         size, tipLength=0.35)
    na = cv2.arrowedLine(na, (grid_size-hl,grid_size//2+grid_size), 
                         (grid_size+hl, grid_size//2+grid_size), (230,0,0), 
                         size, tipLength=0.35)
    im = Image.fromarray(na)
    return im


def make_grid(img_A, img_A_prime, img_B, img_B_prime=None, 
              save_path=None, grid_size=256, p=2, overwrite=False, 
              add_marks=False, add_arrows=False, mark_size=36):

    assert p % 2 == 0
    process_size = int(grid_size - p * 1.5)

    img_A = Image.open(img_A) if isinstance(img_A, str) else img_A
    img_A_prime = Image.open(img_A_prime) if isinstance(img_A_prime, str) else img_A_prime
    img_B = Image.open(img_B) if isinstance(img_B, str) else img_B

    img_A = img_A.convert('RGB').resize((process_size, process_size))
    img_A_prime = img_A_prime.convert('RGB').resize((process_size, process_size))
    img_B = img_B.convert('RGB').resize((process_size, process_size))

    image = Image.new('RGB', (grid_size * 2, grid_size * 2), (255, 255, 255))
    image.paste(img_A, (0+p, 0+p))
    image.paste(img_A_prime, (grid_size+p-1, 0+p))
    image.paste(img_B, (0+p, grid_size+p-1))
    
    if img_B_prime:
        img_B_prime = Image.open(img_B_prime) if isinstance(img_B_prime, str) else img_B_prime
        img_B_prime = img_B_prime.convert('RGB').resize((process_size, process_size))
    else:
        img_B_prime = Image.new('RGB', (process_size, process_size), (0, 0, 0))
    image.paste(img_B_prime, (grid_size+p-1, grid_size+p-1))

    if add_marks:
        image = add_marks_to_grid(image, grid_size, p, size=mark_size)
    if add_arrows:
        image = add_arrows_to_grid(image, grid_size)

    if save_path is None:
        return image
    
    if (not os.path.exists(save_path)) or overwrite:
        image.save(save_path)
    else:
        raise ValueError(f"save_path {save_path} is None or already exists")
    
    return image


def prepare_grid_images(img_A, img_Ap, img_B, res, output_dir): 
    image_fn = os.path.join(output_dir, "input.png")
    make_grid(img_A, img_Ap, img_B, save_path=image_fn, overwrite=True, grid_size=res//2, p=2)
    image_gpt4v_fn = os.path.join(output_dir, "input_marked.png")
    make_grid(img_A, img_Ap, img_B, save_path=image_gpt4v_fn, overwrite=True, grid_size=res//2, p=2, 
              add_marks=True, add_arrows=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visual Prompting')
    parser.add_argument('--img_A', type=str, default="example/caricature_raw/001.png")
    parser.add_argument('--img_Ap', type=str, default="example/caricature_raw/002.png")
    parser.add_argument('--img_B', type=str, default="example/caricature_raw/003.png")
    parser.add_argument('--output_dir', type=str, default="example/caricature_processed")
    parser.add_argument('--res', type=int, default=512)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prepare_grid_images(args.img_A, args.img_Ap, args.img_B, args.res, args.output_dir)

