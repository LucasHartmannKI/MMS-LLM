# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: June 23, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================
import requests
import json
import os
import openai
import pickle
import torch
import clip
from PIL import Image
from torch.nn import CosineSimilarity
import csv
import argparse
from functools import reduce

parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", type=str, default='./example_material')
parser.add_argument('--openai_api_key', type=str, required=True)
parser.add_argument('--gpt_type', type=str, default='gpt4', choices=['gpt4', 'gpt3.5', 'gpt-3.5-turbo'])
parser.add_argument('--api_url', type=str, required=True, help="Custom API URL for GPT requests.") 
args = parser.parse_args()

# set up API key
openai.api_key = args.openai_api_key

# set up CLIP
cos = CosineSimilarity(dim=1, eps=1e-6)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define API request function
def send_gpt_request(api_url, api_key, model, text):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer no-key" 
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "top_p": 0.2
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            print("Response:", response.json())
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "No content available.")
        else:
            print("Failed with status code:", response.status_code)
            return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Summarize captions using GPT
def summarize_captions(text, gpt_type):
    return send_gpt_request(args.api_url, args.openai_api_key, gpt_type, f"Given a set of descriptions about the same 3D object, distill these descriptions into one concise caption. The descriptions are as follows: '{text}'. Avoid describing background, surface, and posture. The caption should be:")

# Load captions for each view (8 views in total)
caps = []
for i in range(8):
    caps.append(pickle.load(open(os.path.join(args.parent_dir, 'Cap3D_captions', f'Cap3D_captions_view{i}.pkl'), 'rb')))
names = []
for i in range(8):
    names.append(set([name.split('-')[0] for name in caps[i].keys()]))
uids = list(reduce(set.intersection, names))

# Change 'w' to 'a' if you want to append to an existing csv file
output_csv = open(os.path.join(args.parent_dir, 'Cap3D_captions', 'Cap3d_captions_final.csv'), 'w')
writer = csv.writer(output_csv)

# Batch processing configuration
batch_size = 100  # Number of UIDs to process in one batch
num_batches = (len(uids) + batch_size - 1) // batch_size  # Total number of batches

print('############begin to generate final captions############')
for batch_idx in range(num_batches):
    batch_start = batch_idx * batch_size
    batch_end = min(batch_start + batch_size, len(uids))
    batch_uids = uids[batch_start:batch_end]

    print(f"Processing batch {batch_idx + 1}/{num_batches}, UIDs {batch_start} to {batch_end - 1}")

    for i, cur_uid in enumerate(batch_uids):
        cur_captions = []
        cur_final_caption = ''
        # For each view, choose the caption with the highest similarity score
        for k in range(8):
            image = preprocess(Image.open(os.path.join(args.parent_dir, 'Cap3D_imgs', f'Cap3D_imgs_view{k}', f'{cur_uid}-{k}.png'))).unsqueeze(0).to(device)
            cur_caption = caps[k][f'{cur_uid}-{k}']
            text = clip.tokenize(cur_caption).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)
            score = cos(image_features, text_features)
            if k == 8 - 1:
                cur_final_caption += cur_caption[torch.argmax(score)]
            else:
                cur_final_caption += cur_caption[torch.argmax(score)] + ', '

        # Summarize captions
        max_retries = 5
        retries = 0
        while retries < max_retries:
            summary = summarize_captions(cur_final_caption, args.gpt_type)
            if 'An error occurred' not in summary:
                break
            retries += 1
        
        if retries == max_retries:
            summary = "Error: Failed to generate summary after multiple retries."

        print(f"{i + batch_start}/{len(uids)} - UID: {cur_uid}, Summary: {summary}")

        # Write result to CSV
        writer.writerow([cur_uid, summary])

    # Flush the file to save progress
    output_csv.flush()
    os.fsync(output_csv.fileno())

# Close the CSV file
output_csv.close()
print("Processing completed.")
