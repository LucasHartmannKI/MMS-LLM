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
parser.add_argument("--parent_dir", type = str, default='./example_material')
parser.add_argument('--openai_api_key', type = str, required = True)
parser.add_argument('--gpt_type', type = str, default='gpt4', choices=['gpt4', 'gpt3.5','gpt-3.5-turbo'])
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
        "model": "gpt-3.5-turbo",
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
#Given a set of descriptions with  different views about the same 3D object, which may be right or wrong, distill these descriptions into one concise caption, using a single concise sentence. The descriptions are as follows: '{text}'.
#"Given a set of descriptions about the same 3D object, which may be right or wrong, distill these descriptions into one concise caption. The descriptions are as follows: '{text}'. Hint: This 3D object is from the street. Your response must focus only on the specific characteristics of the object itself, such as its type, details, or any unique features (e.g., graffiti), ignoring background, color and posture, using a single concise sentence. The caption should be:"
#Your response must focus only on the specific characteristics of the object itself, ignoring background, color and posture, using a single concise sentence. 
# load captions for each view (8 views in total)
caps = []
for i in range(8):
    caps.append(pickle.load(open(os.path.join(args.parent_dir,'Cap3D_captions','Cap3D_captions_view'+str(i)+'.pkl'), 'rb')))
#print(f"caps: {caps}")
names = []
for i in range(8):
    #names.append(set(caps[i].keys())) 
    names.append(set([name.split('-')[0] for name in caps[i].keys()]))
#print(f"Names from each view: {names}")
uids = list(reduce(set.intersection, names))
#print(f"UIDs: {uids}")

# please remove existing uids 

# change 'w' to 'a' if you want to append to an existing csv file
output_csv = open(os.path.join(args.parent_dir, 'Cap3D_captions', 'Cap3d_captions_final.csv'), 'w')
writer = csv.writer(output_csv)

print('############begin to generate final captions############')
for i, cur_uid in enumerate(uids):
    cur_captions = []
    cur_final_caption = ''
    # for each view, choose the caption with the highest similarity score
    # run 8 times to get 8 captions
    for k in range(8):
        image = preprocess(Image.open(os.path.join(args.parent_dir, 'Cap3D_imgs', 'Cap3D_imgs_view%d'%k, cur_uid + '-%d.png'%k))).unsqueeze(0).to(device)
        cur_caption = caps[k][cur_uid+'-%d'%k]
        text = clip.tokenize(cur_caption).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        score = cos(image_features, text_features)
        if k == 8-1:
            cur_final_caption += cur_caption[torch.argmax(score)]
        else:
            cur_final_caption += cur_caption[torch.argmax(score)] + ', '
    #print(f"cur_final_caption:{cur_final_caption}")
    # sometimes, OpenAI API will return an error, so we need to try again until it works
    max_retries = 5
    retries = 0
    while retries < max_retries:
        summary = summarize_captions(cur_final_caption, args.gpt_type)
        #print(f"summary:{summary}")
        if 'An error occurred' not in summary:
            break
        retries += 1
    
    if retries == max_retries:
        summary = "Error: Failed to generate summary after multiple retries."

    
    print(i, cur_uid, summary)

    # write to csv
    writer.writerow([cur_uid, summary])
    if (i)% 1000 == 0:
        output_csv.flush()
        os.fsync(output_csv.fileno())

output_csv.close()


