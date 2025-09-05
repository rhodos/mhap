import pandas as pd
import numpy as np
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from huggingface_hub import login

import code.utils as utils

preamble = f"""
You are a mental health app classifier. Your task is to categorize an app (based on its title and description) as either 'Mental Health' or 'Not Mental Health'. Apps involving talk therapy, coaching, mood tracking, or CBT training should be categorized as 'Mental Health'. You can also look for keywords and phrases like 'Cognitive Behavioral Therapy', 'Mental Well-Being', etc. as indicators that they should be labeled as 'Mental Health'.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_data_point(data, index):
    return dict(data.iloc[index])

def get_first_k_lines(text, k):
    lines = text.split("\n")
    return " ".join(lines[:k])

def construct_example_for_prompt(example_data, i):
    data_point = get_data_point(example_data, i)
    title = data_point["title"]
    description = get_first_k_lines(data_point["description"], k=4)
    category = data_point["label"]
    prompt = f"""
  EXAMPLE {i+1}:
    INPUT: {title}: {description}
    OUTPUT: {category}
  """
    return prompt

def generate_example_i(example_data, i):
    example_i = (
        f"EXAMPLE {i}: {construct_example_for_prompt(get_data_point(example_data, i))}"
    )
    return example_i

def construct_prompt(
    data_point, example_data, preamble=preamble, num_examples=4, max_tokens=300
):
    title = data_point["title"]
    description = get_first_k_lines(data_point["description"], k=4)
    example_str = "\n".join(
        [construct_example_for_prompt(example_data, i) for i in range(num_examples)]
    )

    prompt = f"""{preamble} Here are {num_examples} examples with the correct category shown:

    ---
    {example_str}
    ---

    Now, classify the following app. Think it through step-by-step, and then output your final answer on the last line. Your final answer should be either 'Mental Health' or 'Not Mental Health', with no apostrophes and no 'OUTPUT'. Use at most {max_tokens} words.

    ---

    INPUT: {title}: {description}
    """
    return prompt

def prompt_model(prompt, max_tokens=10, device=device, verbose=True):
    if verbose:
        print("PROMPT:\n", prompt)
    inputs = tokenizer(prompt, return_tensors="pt")

    if verbose:
        print("LENGTH OF INPUT:", len(inputs["input_ids"][0]))

    # move input tensors to GPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    encoded_output = model.generate(
        **inputs, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id
    )

    # Decode only the newly generated tokens
    output = tokenizer.decode(
        encoded_output[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
    )

    if verbose:
        print("MODEL:\n", output)
    return output

if __name__ == "__main__":

    # Load the data
    data_dir = utils.get_data_dir(step=3)
    data_file = os.path.join(data_dir, "validation_data.tsv")

    data = pd.read_csv(data_file, sep="\t")
    data = data[["title", "description", "label"]]

    # Extract 30 samples to use as test data, 4 for in-context examples
    sampled_data = data.sample(34)
    test_data = sampled_data[:30]
    example_data = sampled_data[30:]
    print(test_data["label"].value_counts())
    print(example_data["label"].value_counts())

    # Log in to Hugging Face using the token from environment variables
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(HF_TOKEN)
        print("Successfully logged in to Hugging Face!")
    else:
        print("HF_TOKEN secret is not set. Please add it to your environment variables.")

    # model_name='google/flan-t5-base'
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = 'mistralai/Mistral-7B-v0.1'
    # model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Move the model to the GPU if available 
    model.to(device)

    output = prompt_model("Hello, how are you?", max_tokens=5, verbose=True)
    print("OUTPUT:", output)

    max_tokens = 300
    outputs = []
    labels = []
    is_correct = []
    for i in range(3):  # range(len(test_data)):
        print("DATA POINT ", i)
        data_point = get_data_point(test_data, i)
        label = data_point["label"]
        prompt = construct_prompt(data_point, example_data, max_tokens=max_tokens)
        output = prompt_model(prompt, max_tokens=max_tokens, verbose=False).strip()
        print("OUTPUT:", output)
        print("TRUE LABEL:", label)
        outputs.append(output)
        labels.append(label)
        is_correct.append(output == label)
        print("\n\n")
