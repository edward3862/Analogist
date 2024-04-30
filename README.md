# Analogist (SIGGRAPH 2024)

This repository is the official implementation of Analogist.

![framework](assets/images/teaser.jpg)

> **Analogist: Out-of-the-box Visual In-Context Learning with Image Diffusion Model.**
> 
> Zheng Gu, Shiyuan Yang, Jing Liao, Jing Huo, Yang Gao.

The code will be coming soon.

## Set Up

First create the conda environment
```
conda create --name analogist python==3.10.0
conda activate analogist
```

Then install the additional requirements
```
pip install -r requirements.txt
```

## Run Analogist

### Visual Prompting

First prepare the input grid image for visual prompting:

```
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

```
export OPENAI_API_KEY="your-api-key-here"
```

Or you can add it directly in `gpt4_prompting.py`:

```
api_key = os.environ.get('OPENAI_API_KEY', "your-api-key-here")
```

Then ask GPT-4V to get the text prompts for $B'$:

```
python textual_prompting.py \
    --image_path example/colorization_processed/input_marked.png \
    --output_prompt_file example/colorization_processed/prompts.txt
```

You can ask GPT-4V to generate several prompts in one request. The text prompts will be saved in `prompts.txt`, with each line shows one text prompt.

For those who do not have `OPENAI_API_KEY`, you can try publicly available vision-language models([MiniGPT](https://minigpt-4.github.io/), [LLaVA](https://llava-vl.github.io/), et al.) with `input_marked.png` as visual input and the following prompt as textual input:

> Please help me with the image analogy task: take an image A and its transformation A’, and provide any image B to produce an output B’ that is analogous to A’. Or, more succinctly: A : A’ :: B : B’. You should give me the text prompt of image B’ with no more than 5 words.

### Run the Diffusion Inpainting Model

After getting the visual prompt and textual prompt, we can take them together to run the [diffusion inpainting model](https://huggingface.co/runwayml/stable-diffusion-inpainting).

```
python analogist.py \
    --input_grid example/colorization_processed/input.png \
    --prompt_file example/colorization_processed/prompts.txt \
    --output_dir output/colorization
```

## Evaluation Datasets

We have prepared all the images used for quantitative evaluation in this [link](https://portland-my.sharepoint.com/:f:/g/personal/zhenggu4-c_my_cityu_edu_hk/Eh_jT6A5s6VHo7Q4GiDAKY4BjqQ3_f9MJ89qdsIEbe_K2g?e=kJY6HB). Download them and put them in a `datasets` folder.

```
.Analogist
├── datasets
│   ├── low_level_tasks_processed
│   │  ├── ... 
│   │  ├── ...
│   ├── manipulation_tasks_processed
│   │  ├── ...
│   │  ├── ...
│   ├── vision_tasks_processed
│   │  ├── ...
│   │  ├── ...
```


## Acknowledgement

We borrow the code of attention control from [Prompt-to-Prompt](https://prompt-to-prompt.github.io/) and [MasaCtrl](https://ljzycmd.github.io/projects/MasaCtrl/).

