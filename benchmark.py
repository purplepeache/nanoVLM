import random
import re
import argparse

from collections import Counter

from tqdm import tqdm

from datasets import load_dataset

import numpy as np

import torch
from PIL import Image
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

args_parser = argparse.ArgumentParser(
    description="A script to process models by dropping layers."
)

args_parser.add_argument(
    '--n_drops',
    type=int,
    default=0,
    help='Number of layers to drop.'
)

args_parser.add_argument(
    '--split',
    type=str,
    default="validation",
    help='Split to benchmark.',
    choices=["train", "validation"],
)

args_parser.add_argument(
    '--n_sample',
    type=int,
    default=-1,
    help='Number of samples to benchmark.',
)

# 3. Parse the arguments
args = args_parser.parse_args()


klein_blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "white_to_klein", 
    ["white", "#002FA7"]
)

# Flatten list function
def flatten_list(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


from inference import NanoVLMInference, get_enclosed_letter

# Initialize the inference wrapper
MODEL_NAME = "lusxvr/nanoVLM"
DEVICE = "cuda"
model = NanoVLMInference(model_name=MODEL_NAME, device=DEVICE)


# Load dataset to benchmark
D = load_dataset("HuggingFaceM4/A-OKVQA")

total_layers = len(model.model.decoder.blocks)
n_drops = args.n_drops
for _ in range(n_drops):
    del model.model.decoder.blocks[-1]
remaining_layers = len(model.model.decoder.blocks)
print(f"Kept {remaining_layers} / {total_layers} layers")

# Doing the benchmark
split = args.split
greedy = True
max_new_tokens = 5

if args.n_sample != -1:
    data_idxs = np.random.choice(len(D[split]), size=args.n_sample, replace=False)
else:
    data_idxs = range(len(D[split]))

outputs = {
    "data_idx": [],
    "question": [],
    "prompt": [],
    "response": [],
    "correct_choice": [],
    "correct_choice_alphabet": [],
}
for data_idx in tqdm(data_idxs):
    # Fetch the datapoint
    data = D[split][data_idx]
    question = data["question"]
    image = data["image"]
    choices = data["choices"]
    correct_choice_idx = data["correct_choice_idx"]
    correct_choice = choices[correct_choice_idx]
    correct_choice_alphabet = chr(ord("A") + correct_choice_idx)

    outputs["data_idx"].append(data_idx)
    outputs["question"].append(question)
    outputs["correct_choice"].append(correct_choice)
    outputs["correct_choice_alphabet"].append(correct_choice_alphabet)

    # Format the prompt
    prompt = f"""
    {question}\n
    """.strip()
    for i, choice in enumerate(choices):
        prompt += f"\n{chr(ord('A') + i)}. {choice}"
    prompt += "\n\n(You must answer with only the letter enclosed in curly bracket, e.g. {A})"

    outputs["prompt"].append(prompt)

    # Generate the response
    response = model.generate_from_image_and_prompt(image=image, prompt=prompt, max_new_tokens=max_new_tokens, greedy=greedy)
    
    outputs["response"].append(response)
    # break

# Parse responses
outputs["parsed_response"] = []
for response_idx, response in enumerate(outputs["response"]):
    if (len(response) == 1) and (response.isalpha()):
        outputs["parsed_response"].append(response.capitalize())
    else:
        parsed_response = get_enclosed_letter(response)
        parsed_response = parsed_response.capitalize() if parsed_response is not None else [i for i in ["A", "B", "C", "D"] if i != outputs["correct_choice_alphabet"][response_idx]][0]
        outputs["parsed_response"].append(parsed_response)


# Measure performance
accuracy = accuracy_score(outputs["correct_choice_alphabet"], outputs["parsed_response"])


# Save the outputs
pd.DataFrame(outputs).to_csv(f"exps/outputs-ZS-{split}-n_drops={n_drops}-n_sample={args.n_sample}.csv", index=False)


print("--------------------------------")
print(f"Model: {MODEL_NAME}")
print(f"Split: {split}")
print(f"Dropped Layers: {n_drops}")
print(f"Accuracy: {accuracy:.3f}")