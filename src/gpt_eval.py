import json
from openai import OpenAI
import re
from tqdm import tqdm
import os
import time

API_BASE = "https://api.openai.com/v1"
API_KEY = ""


client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

system = [
        "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to a query taken from the test dataset displayed below.",
        "A ground truth answer is provided and should be treated as the correct reference.",
        "Assess whether the assistant’s response is accurate compared to the ground truth and whether the wording and explanation are appropriate and coherent.",
        "Begin your evaluation by providing a short explanation. Be as objective as possible.",
        'After providing your explanation, please rate the response on a scale of 1 to 100 by strictly following this format: "[[rating]]",",',
        'for example: "Rating: [[90]]"."'
    ]

system_prompt = """[System]
{}
[Question]
{}
[Ground truth answer]
{}

[The Start of Assistant’s Answer]
{}
[The End of Assistant’s Answer]"""

def gpt_evaluate(ground_truth_path, eval_path):
    with open(ground_truth_path) as f:
        ground_truth = [json.loads(l) for l in f.readlines()]

    with open(eval_path) as f:
        eval_data = [json.loads(l) for l in f.readlines()]

    for i, item in enumerate(eval_data):
        item['gt_answer'] = ground_truth[i]['output']

    gpt_responses = []

    for item in tqdm(eval_data):
        content = system_prompt.format(' '.join(system), item["prompt"], item['gt_answer'], item['generated_text'])
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            # model="gpt-5-nano",
            messages=[{"role": "user", "content": content}],
            temperature=0.7,
        )

        gpt_response = completion.choices[0].message.content

        gpt_responses.append({
            "prompt": item["prompt"],
            "gpt_response": gpt_response
        })

    sum_rating = 0
    for item in gpt_responses:
        response = item['gpt_response']
        matches = re.findall(r"\[\[\s*(\d+)\s*\]\]", response)
        rateing = int(matches[-1])
        sum_rating += rateing

    avg_rating = sum_rating / len(gpt_responses)
    print(eval_path)
    print(f"Average rating: {avg_rating}")

    return avg_rating, gpt_responses



folder_path = r"path_to_eval_text"

gt_map = {}
for root, dirs, files in os.walk(folder_path):
    for file in files:
        full_path = os.path.join(root, file)
        # print(full_path)
        if 'dolly' in full_path:
            gt_map[full_path] = "data/dolly/valid.jsonl"
        elif 'vicuna' in full_path:
            gt_map[full_path] = "data/vicuna/valid.jsonl"
        elif 'sni' in full_path:
            gt_map[full_path] = "data/sinst/11_/valid.jsonl"
        elif 'self_instruct' in full_path:
            gt_map[full_path] = "data/self-inst/valid.jsonl"


# gt_map

def main():
    for k, v in gt_map.items():
        try:
            avg_rating, gpt_responses = gpt_evaluate(v, k)
        except:
            time.sleep(600)
            avg_rating, gpt_responses = gpt_evaluate(v, k)

        result_file_name = ""
        if 'dolly' in k:
            result_file_name = "dolly"
        elif 'vicuna' in k:
            result_file_name = "vicuna"
        elif 'sni' in k:
            result_file_name = "sni"
        elif 'self_instruct' in k:
            result_file_name = "self_instruct"
        

        parent_dir = 'gpt_eval/' + '/'.join(k.split('/')[1:-1])
        results = {
            'avg_rating': avg_rating,
            'gpt_responses': gpt_responses
        }
        os.makedirs(parent_dir, exist_ok=True)
        with open(parent_dir + f"/{result_file_name}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

if __name__ == "__main__":
    main()