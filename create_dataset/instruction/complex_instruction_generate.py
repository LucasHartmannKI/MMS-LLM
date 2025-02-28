import base64
import json
from openai import OpenAI
import os

def encode_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def format_response(response, image_path):
    object_id = os.path.splitext(os.path.basename(image_path))[0]

    try:
        # Clean and parse the JSON content from the response
        raw_content = response.choices[0].message.content
        json_content = raw_content.strip("```json\n").strip("```").strip()
        api_content = json.loads(json_content)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON content from response: {e}")
    
    # Create the detailed description format
    formatted_output = [
        {
            "object_id": object_id,
            "conversation_type": "detailed_description",
            "conversations": [
                {
                    "from": "human",
                    "value": "<point>\nProvide a meticulous explanation of what these points represent."
                },
                {
                    "from": "gpt",
                    "value": api_content["caption"]
                }
            ]
        }
    ]

    # Add single-round conversations
    single_round_conversations = api_content["single_conversation"]
    for convo in single_round_conversations:
        formatted_output.append({
            "object_id": object_id,
            "conversation_type": "single_round",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<point>\n{convo['Q']}"
                },
                {
                    "from": "gpt",
                    "value": convo["A"]
                }
            ]
        })

    # Add multi-round conversations
    multi_round_conversation = api_content["multi_conversation"][0]
    formatted_output.append({
        "object_id": object_id,
        "conversation_type": "multi_round",
        "conversations": [
            {
                "from": "human",
                "value": f"<point>\n{multi_round_conversation['Q1']}"
            },
            {
                "from": "gpt",
                "value": multi_round_conversation["A1"]
            },
            {
                "from": "human",
                "value": multi_round_conversation["Q2"]
            },
            {
                "from": "gpt",
                "value": multi_round_conversation["A2"]
            },
            {
                "from": "human",
                "value": multi_round_conversation["Q3"]
            },
            {
                "from": "gpt",
                "value": multi_round_conversation["A3"]
            }
        ]
    })

    return formatted_output

def make_api_request(image_path, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url='your_base_url' # Here you can specify the base URL for the API
    )
    
    try:
        encoded_image = encode_image(image_path)
        
        response = client.chat.completions.create(
            model=client.models.list().data[0].id,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze and describe 3D models accurately."#You are a helpful assistant that can see and describe images.
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": """
The 3D point cloud object that needs to be analyzed is the object surrounded by the green circle in the image. Do not reference any 'image', figure, or visual elements. \
Focus entirely on the object represented by the 3D model without mentioning any color, figure, green line, or image context:

1. Write a new detailed caption (50-100 words) describing its type, appearance（without color）, functionality, or daily-life usage.
2. Generate 3 single Q&As about the object based on the captions.
3. Create 1 set of logically connected 3-round Q&As.

Respond in JSON:
{
  "caption": "description",
  "single_conversation": [{"Q": "Q", "A": "A"} x 3],
  "multi_conversation": [{"Q1": "Q", "A1": "A", "Q2": "Q", "A2": "A", "Q3": "Q", "A3": "A"} x 1]
}
"""},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response
    
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        raise

def process_images(image_folder, api_key, output_json):
    """Process all images in a folder and generate a JSON output."""
    results = []

    for root, _, files in os.walk(image_folder):
        for file in files:
            try:
                image_path = os.path.join(root, file)
                print(f"Processing image: {file}")
                response = make_api_request(image_path, api_key)
                # Format the response
                formatted_data = format_response(response, image_path)             
                results.extend(formatted_data)  # Add each image's result to the list
            except Exception as e:
                print(f"Failed to process image {file}: {e}")

    # Save the results to a JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_json}")



if __name__ == '__main__':
    # Define paths and API key
    image_folder = "data/large/Cap3D/captioning_pipeline/example_material/Cap3D_imgs/image_original_filter2000" #image_original_filter
    api_key = "YOUR_API_KEY"
    output_json = "/home/PointLLM/data/anno_data/ikgc17_complex_descriptions.json"

    # Process images and save the results
    process_images(image_folder, api_key, output_json)