import os
import base64
import requests
import argparse

# OpenAI API Key
api_key = os.environ.get('OPENAI_API_KEY', None)

# Prompt
PROMPT = "Please help me with the image analogy task: take an image A and its transformation A', and provide any image B to produce an output B' that is analogous to A'. Or, more succinctly: A : A' :: B : B'. You should give me the most possible text prompt of image B' with no more than 5 words."

# PROMPT = "Please help me with the image analogy task: take an image A and its transformation A', and provide any image B to produce an output B' that is analogous to A'. Or, more succinctly: A : A' :: B : B'. You should give me three most possible text prompts of image B', each prompt should contain no more than 5 words. Please separate the prompts with comma and format your answer as follows: prompt1, prompt2, prompt3"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def ask_gpt4(image_path, prompt=PROMPT):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def postprocess_gpt4_response(response, sep=","):
    # Parsing the response
    prompts = response["choices"][0]["message"]["content"]
    prompts = prompts.strip().split(sep)
    prompts = [p.strip() for p in prompts]
    return prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Textual Prompting')
    parser.add_argument('--image_path', type=str, default="example/colorization_processed/input_marked.png")
    parser.add_argument('--output_prompt_file', type=str, default="example/colorization_processed/prompts.txt")
    args = parser.parse_args()
    
    prompt = "Please help me with the image analogy task: take an image A and its transformation A', and provide any image B to produce an output B' that is analogous to A'. Or, more succinctly: A : A' :: B : B'. You should give me three most possible text prompts of image B', each prompt should contain no more than 5 words. Please separate the prompts with comma and format your answer as follows: prompt1, prompt2, prompt3"

    gpt4_answer = ask_gpt4(args.image_path, prompt=prompt)
    prompts = postprocess_gpt4_response(gpt4_answer)
    print(prompts)
    with open(args.output_prompt_file, "w") as f:
        f.write("\n".join(prompts))