# Analogist (SIGGRAPH 2024)

This repository is the official implementation of Analogist.

![framework](assets/images/teaser.jpg)

> **Analogist: Out-of-the-box Visual In-Context Learning with Image Diffusion Model.**
> 
> Zheng Gu, Shiyuan Yang, Jing Liao, Jing Huo, Yang Gao.


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

It is recommended for users to obtain an available OpenAI API and adding it into your environment:

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

Here is another example of translating a photo into caricature.

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

The other things are the same. Note that we use the same value of hyper-parameters in the quantitative evaluation. However, it is recommended to try different combinations in specific cases for better results.

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

