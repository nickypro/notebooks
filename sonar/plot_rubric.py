# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_score_for_each_method(file_path, metric):
    with open(file_path, "r") as f:
        data = json.load(f)
    print(data)

    # Example data : {"index": 0, "reference": "**Black Seed Oil: A Comprehensive Review**\n\n", "comparison": "**Problem 1**\nA 30-year-old male patient is diagnosed with a 4 cm diameter, 2 cm thick, and 5 cm long tumor in the right kidney. The tumor is classified as a renal cell carcinoma (RCC). The patient has a history of hypertension and is currently taking beta blockers and an ACE inhibitor. The patient has a family history of RCC. The patient's serum creatinine level is 1.2 mg/dL, and the estimated glomerular filtration rate (eGFR) is 60 mL/min/1.73 m\\({}^{2}\\). The patient is a non-sm", "result": "{\"reasoning\": {\"complexity\": \"Text 1 presents a title indicating a review, while Text 2 provides a detailed clinical case. Thus: many details in Text 2, but trivial in Text 1: thus out of 3, score 1.\", \"coherence\": \"Text 2 presents a coherent clinical scenario, while Text 1 is just a title. Thus: Text 2 is mostly coherent with minor errors, while Text 1 is trivial: thus out of 3, score 2.\", \"structure\": \"Text 1 is a title, while Text 2 is a detailed paragraph. Thus: no alignment in structure: thus out of 2, score 0.\", \"subject\": \"Text 1 discusses Black Seed Oil, while Text 2 discusses a clinical case of renal cell carcinoma. Thus: completely unrelated subjects: thus out of 4, score 0.\", \"entities\": \"Text 1 does not mention specific entities, while Text 2 includes entities related to a medical case. Thus: no entities to compare: thus out of 4, score -1.\", \"details\": \"Text 1 lacks details, while Text 2 provides specific clinical information. Thus: details differ completely: thus out of 3, score 0.\", \"terminology\": \"Text 1 does not contain medical terminology, while Text 2 uses specific medical terms. Thus: no shared terms: thus out of 2, score 0.\", \"tone\": \"Text 1 has a neutral informative tone, while Text 2 maintains a clinical tone. Thus: consistent tone: thus out of 1, score 1.\"}, \"scoring\": {\"complexity\": 1, \"coherence\": 2, \"structure\": 0, \"subject\": 0, \"entities\": -1, \"details\": 0, \"terminology\": 0, \"tone\": 1}}", "type": "cheat-1"}

def get_scores(data_list):
    # Convert string JSON to dict if needed
    scores = []
    for item in data_list:
        if isinstance(item['result'], str):
            result = json.loads(item['result'])
        else:
            result = item['result']
        # score = result['scoring']
        scores.append(result["scoring"])
    return scores


def process_scores(data_list, metric):
    # Convert string JSON to dict if needed
    scores = []
    for item in get_scores(data_list):
        # if item["coherence"] < 2:
        #     continue
        score = item[metric]
        if score != -1:
            scores.append(score)
    return scores

def calculate_score_proportions(scores, cumulative=False):
    total = len(scores)
    if total == 0:
        return []

    # Get unique possible scores and sort them
    unique_scores = sorted(set(scores))
    proportions = []

    if cumulative:
        # Calculate proportion >= each score
        for threshold in unique_scores:
            count = sum(1 for score in scores if score >= threshold)
            proportions.append(count / total)
    else:
        # Calculate proportion = each score
        for score in unique_scores:
            count = sum(1 for s in scores if s == score)
            proportions.append(count / total)

    return proportions

def plot_score_proportions(compare_files, metric, output_image=None):
    plt.figure(figsize=(12, 6))

    # Process each comparison type
    for label, file_path in compare_files.items():
        # Load and process data
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in f]

        scores = process_scores(data, metric)
        proportions = calculate_score_proportions(scores)

        # Get unique scores for x-axis
        unique_scores = sorted(set(scores))

        # Plot as lines
        # Plot stacked bars for each score threshold
        # Create a base color for this label using a consistent mapping
        label_index = list(compare_files.keys()).index(label)
        base_color = plt.cm.Pastel1(label_index / len(compare_files))

        bottom = 0
        for i, prop in reversed(list(enumerate(proportions))):
            # Darken the base color based on score level
            darkness = 1 - (i/len(proportions))
            color = tuple(c * darkness for c in base_color[:3]) + (base_color[3],)

            bar = plt.bar([label], [prop], bottom=bottom,
                   label=f'{label} (≥{unique_scores[i]})',
                   color=color)
            plt.text(bar[0].get_x() + bar[0].get_width()/2, bottom + prop/2,
                    str(unique_scores[i]),
                    ha='center', va='center')
            bottom += prop

    plt.xlabel(f'{metric} Score Threshold')
    plt.ylabel('Proportion >= Score')
    plt.title(f'Cumulative Score Distribution for {metric}')
    plt.grid(True, alpha=0.3)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_image:
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    # Manually list the files.
    compare_files = {
        "mlp-train": "processed_rubrics/log5_mlp-train.jsonl",
        "linear-train": "processed_rubrics/log5_linear-train.jsonl",

        "mlp": "processed_rubrics/ruberic_log_4o_best.jsonl_mlp.jsonl",
        "linear": "processed_rubrics/ruberic_log_4o_2.jsonl_linear.jsonl",
        "continued": "processed_rubrics/log4.txt_continued.jsonl",
        "baseline": "processed_rubrics/log4.txt_baseline.jsonl",
        "cheat-1": "processed_rubrics/log4.txt_cheat-1.jsonl",
        "cheat-5": "processed_rubrics/log4.txt_cheat-5.jsonl",
        "cheat-10": "processed_rubrics/log4.txt_cheat-10.jsonl",
        "regenerated": "processed_rubrics/log4.txt_regenerated.jsonl",
        "auto-decoded": "processed_rubrics/log4.txt_auto-decoded.jsonl"
    }

    metrics = ["complexity", "coherence", "structure", "subject", "entities", "details", "terminology", "tone"]
    for metric in metrics:
        plot_score_proportions(compare_files, metric, f"{metric}.png")

# %%
