# Analogist (SIGGRAPH 2024)

This repository is the official implementation of Analogist.

![framework](assets/images/teaser.jpg)

> **Analogist: Out-of-the-box Visual In-Context Learning with Image Diffusion Model.**
> 
> Zheng Gu, Shiyuan Yang, Jing Liao, Jing Huo, Yang Gao.
>
>Visual In-Context Learning (ICL) has emerged as a promising research area due to its capability to accomplish various tasks with limited example pairs through analogical reasoning. However, training-based visual ICL has limitations in its ability to generalize to unseen tasks and requires the collection of a diverse task dataset. On the other hand, existing methods in the inference-based visual ICL category solely rely on textual prompts, which fail to capture fine-grained contextual information from given examples and can be time-consuming when converting from images to text prompts. To address these challenges, we propose Analogist, a novel inference-based visual ICL approach that exploits both visual and textual prompting techniques using a text-to-image diffusion model pretrained for image inpainting. For visual prompting, we propose a self-attention cloning (SAC) method to guide the fine-grained structural-level analogy between image examples. For textual prompting, we leverage GPT-4V's visual reasoning capability to efficiently generate text prompts and introduce a cross-attention masking (CAM) operation to enhance the accuracy of semantic-level analogy guided by text prompts. Our method is out-of-the-box and does not require fine-tuning or optimization. It is also generic and flexible, enabling a wide range of visual tasks to be performed in an in-context manner. Extensive experiments demonstrate the superiority of our method over existing approaches, both qualitatively and quantitatively.


## Set Up

First create the conda environment

```bash
conda create --name analogist python==3.10.0
conda activate analogist
```

Then install the additional requirements

```bash
pip install -r requirements.txt
```

## Run Analogist

### Visual Prompting

First prepare the input grid image for visual prompting:

```bash
python visual_prompting.py \
    --img_A example/colorization_raw/001.png \
    --img_Ap example/colorization_raw/002.png \
    --img_B example/colorization_raw/003.png \
    --output_dir example/colorization_processed
```

This will generate two images under the `output_dir` folder:
- `input.png`: $2\times2$ grid image consisting of $A$, $A'$, and $B$.
- `input_marked.png`: edited grid image with marks and arrows added.

### Textual Prompting

It is recommended to obtain an available OpenAI API and adding it into your environment:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or you can add it directly in `gpt4_prompting.py`:

```python
api_key = os.environ.get('OPENAI_API_KEY', "your-api-key-here")
```

Then ask GPT-4V to get the text prompts for $B'$:

```bash
python textual_prompting.py \
    --image_path example/colorization_processed/input_marked.png \
    --output_prompt_file example/colorization_processed/prompts.txt
```

You can ask GPT-4V to generate several prompts in one request. The text prompts will be saved in `prompts.txt`, with each line shows one text prompt.

For those who do not have `OPENAI_API_KEY`, you can try publicly available vision-language models([MiniGPT](https://minigpt-4.github.io/), [LLaVA](https://llava-vl.github.io/), et al.) with `input_marked.png` as visual input and the following prompt as textual input:

> Please help me with the image analogy task: take an image A and its transformation A’, and provide any image B to produce an output B’ that is analogous to A’. Or, more succinctly: A : A’ :: B : B’. You should give me the text prompt of image B’ with no more than 5 words.

### Run the Diffusion Inpainting Model

After getting the visual prompt and textual prompt, we can take them together to run the [diffusion inpainting model](https://huggingface.co/runwayml/stable-diffusion-inpainting).

```bash
python analogist.py \
    --input_grid example/colorization_processed/input.png \
    --prompt_file example/colorization_processed/prompts.txt \
    --output_dir results/example/colorization
```

Here is another example of translating a photo into a caricature. Note that we use the same value of hyper-parameters in the quantitative evaluation. However, it is recommended to try different combinations in specific cases for better results.

```bash
python analogist.py \
    --input_grid example/caricature_processed/input.png \
    --prompt_file example/caricature_processed/prompts.txt \
    --output_dir results/example/caricature \
    --num_images_per_prompt 1 --res 1024 \
    --sac_start_layer 4 --sac_end_layer 9 \
    --cam_start_layer 4 --cam_end_layer 9 \
    --strength 0.96 --scale_sac 1.3 --guidance_scale 15
```

## Datasets

All the image datasets can be achieved through this [link](https://portland-my.sharepoint.com/:f:/g/personal/zhenggu4-c_my_cityu_edu_hk/Eh_jT6A5s6VHo7Q4GiDAKY4BjqQ3_f9MJ89qdsIEbe_K2g?e=kJY6HB). Please put them in a `datasets` folder. We also provide the GPT-4V prompts that we used in our experiments. Please see the `*_gpt4_out.txt` files.

```
Analogist
├── datasets
│   ├── low_level_tasks_processed
│   │  ├── ... 
│   │  ├── *_gpt4_out.txt
│   ├── manipulation_tasks_processed
│   │  ├── ...
│   │  ├── *_gpt4_out.txt
│   ├── vision_tasks_processed
│   │  ├── ...
│   │  ├── *_gpt4_out.txt
```


## More Applications

### $A$ is aligned with $B$ instead of $A'$. 

In this case, the only modification is to swap the position of $A'$ and $B$ in the grid image.

```bash
python visual_prompting.py \
    --img_A example/corgi_raw/001.png \
    --img_Ap example/corgi_raw/003.png \
    --img_B example/corgi_raw/002.png \
    --output_dir example/corgi_processed
```

The other things are the same.

```bash
python analogist.py \
    --input_grid example/corgi_processed/input.png \
    --prompt_file example/corgi_processed/prompts.txt \
    --output_dir results/example/corgi \
    --num_images_per_prompt 1 --res 1024 \
    --sac_start_layer 4 --sac_end_layer 8 \
    --cam_start_layer 4 --cam_end_layer 8 \
    --strength 0.8 --scale_sac 1.3 --guidance_scale 15
```

## Acknowledgement

We borrow the code of attention control from [Prompt-to-Prompt](https://prompt-to-prompt.github.io/) and [MasaCtrl](https://ljzycmd.github.io/projects/MasaCtrl/).

