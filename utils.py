import os
import cv2
import torch
import random
import numpy as np

from PIL import Image
from PIL import ImageDraw, ImageFont


def process_grid(img, p=2, gird_size=256):
    assert p % 2 == 0
    (w, h) = img.size
    img_np = np.array(img)
    p = 2
    num_line_h = h // gird_size
    num_line_w = w // gird_size
    for i in range(num_line_h):
        img_np[gird_size*i-p//2:gird_size*i+p//2, :, :] = 255
    for i in range(num_line_w):
        img_np[:, gird_size*i-p//2:gird_size*i+p//2, :] = 255
    img_np[:p, :, :] = 255
    img_np[:, :p, :] = 255
    img_np[-p:, :, :] = 255
    img_np[:, -p:, :] = 255
    img = Image.fromarray(img_np)
    return img


def add_mark(image, text, coord, font_color=(255, 255, 255), size=48):
    x, y = coord
    width=size
    height=size
    fontsize=size
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arialbd.ttf", fontsize)
    draw.rectangle([x, y, x + width, y + height], fill="black")
    draw.text((x+2, y), text, font=font, fill=font_color)
    return image


def add_marks_to_grid(image, grid_size, p, size=48):
    image = add_mark(image, "A", (0+p, 0+p), size=size)
    image = add_mark(image, "A'", (grid_size+p-1, 0+p), size=size)
    image = add_mark(image, "B", (0+p, grid_size+p-1), size=size)
    image = add_mark(image, "B'", (grid_size+p-1, grid_size+p-1), size=size)
    return image


def add_arrows_to_grid(im, grid_size=256, hl=25, size=5):
    hl = int(grid_size / 10)
    size = int(grid_size / 64)
    na = np.array(im)
    na = cv2.arrowedLine(na, (grid_size-hl,grid_size//2), (grid_size+hl, grid_size//2), (230,0,0), size, tipLength=0.35)
    na = cv2.arrowedLine(na, (grid_size-hl,grid_size//2+grid_size), (grid_size+hl, grid_size//2+grid_size), (230,0,0), size, tipLength=0.35)
    im = Image.fromarray(na)
    return im


def make_grid(img_A, img_A_prime, img_B, img_B_prime=None, save_path=None, grid_size=256, p=2, overwrite=False, add_marks=False, add_arrows=False, mark_size=48):

    assert p % 2 == 0
    process_size = int(grid_size - p * 1.5)

    img_A = Image.open(img_A) if isinstance(img_A, str) else img_A
    img_A_prime = Image.open(img_A_prime) if isinstance(img_A_prime, str) else img_A_prime
    img_B = Image.open(img_B) if isinstance(img_B, str) else img_B

    img_A = img_A.convert('RGB').resize((process_size, process_size))
    img_A_prime = img_A_prime.convert('RGB').resize((process_size, process_size))
    img_B = img_B.convert('RGB').resize((process_size, process_size))

    image = Image.new('RGB', (grid_size * 2, grid_size * 2), (0, 0, 0))
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


def decompose_grid(image, p=2, grid_size=256):
    
    assert p % 2 == 0
    process_size = int(grid_size - p * 1.5)

    img_A = image.crop((0+p, 0+p, process_size+p, process_size+p))
    img_A_prime = image.crop((grid_size+p-1, 0+p, grid_size+process_size+p-1, process_size+p))
    img_B = image.crop((0+p, grid_size+p-1, process_size+p, grid_size+process_size+p-1))
    img_B_prime = image.crop((grid_size+p-1, grid_size+p-1, grid_size+process_size+p-1, grid_size+process_size+p-1))

    img_A = img_A.resize((grid_size, grid_size))
    img_A_prime = img_A_prime.resize((grid_size, grid_size))
    img_B = img_B.resize((grid_size, grid_size))
    img_B_prime = img_B_prime.resize((grid_size, grid_size))

    return img_A, img_A_prime, img_B, img_B_prime


def get_enc_latent(pipeline, image, generator=None, device="cuda", dtype=torch.float16, height=512, width=512):
    if isinstance(image, (Image.Image, np.ndarray)):
        image = [image]
    if isinstance(image, list) and isinstance(image[0], Image.Image):
        image = [i.resize((width, height), resample=Image.LANCZOS) for i in image]
        image = [np.array(i.convert("RGB"))[None, :] for i in image]
        image = np.concatenate(image, axis=0)
    image = image.transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    image = image.to(device=device, dtype=dtype)
    image_latents = pipeline._encode_vae_image(image=image, generator=generator)
    return image_latents


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def img2imgs(fn, w=512, h=512):
    img = Image.open(fn)
    width, height = img.size
    img0 = img.crop((0, 0, w, h))
    imgs = []
    for i in range(1, width // w, 1):
        imgi = img.crop((i*w, 0, (i+1)*w, h)).resize((w//2, h//2))
        grid = Image.new('RGB', (w, h), (0, 0, 0))
        grid.paste(img0, (0, 0))
        grid.paste(imgi, (w//2, h//2))
        imgs.append(grid)
    return imgs