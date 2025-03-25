import pandas as pd
import random
import json

def main():
    questions = [
        '<point>\nDeliver a quick description of the object represented here.',
        '<point>\nGive a brief explanation of the object that this cloud of points forms.',
        '<point>\nWhat object is this point cloud rendering?',
        '<point>\nHow would you interpret this 3D point cloud?',
        '<point>\nCould you delineate the form indicated by this point cloud?',
        '<point>\nDescribe the object that this point cloud.',
        '<point>\nProvide a short explanation of this 3D structure.',
        '<point>\nHow would you describe the 3D form shown in this point cloud?',
        '<point>\nWhat does this collection of points represent?',
        '<point>\nGive a quick overview of the object represented by this 3D cloud.',
        '<point>\nCan you briefly outline the shape represented by these points?',
        '<point>\nWhat can you infer about the object from this point cloud?',
        '<point>\nSummarize the 3D point cloud object briefly.',
        '<point>\nOffer a clear and concise description of this point cloud object.',
        '<point>\nExpress in brief, what this point cloud is representing.',
        '<point>\nOffer a succinct summary of this 3D object.',
        '<point>\nOffer a summary of the 3D object illustrated by this cloud.',
        '<point>\nGive a concise interpretation of the 3D data presented here.'
    ]


    file_path = '/data/large/Cap3D/captioning_pipeline/example_material/Cap3D_captions/Cap3d_captions_final.csv' 
    data = pd.read_csv(file_path)


    json_data = []
    for index, row in data.iterrows():
        object_id = row[0]  
        description = row[1]  # 
        question = random.choice(questions)  # random select question

        json_entry = {
            "object_id": object_id,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": description
                }
            ]
        }
        json_data.append(json_entry)


    output_file = '/home/PointLLM/data/anno_data/ikgc17_brief_description.json'  
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
    print(f"Finished writing to {output_file}")