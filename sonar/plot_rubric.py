# %%
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def plot_score_for_each_method(file_path, metric):
    with open(file_path, "r") as f:
        data = json.load(f)
    print(data)

    # Example data :
    # {"index": 0, "reference": "", "comparison": "", "type": "cheat-1"
    # "result": {
    #     "reasoning": "[reasoning here]"
    #     "scoring": {
    #         "complexity": 1, "coherence": 2, "structure": 0, "subject": 0,
    #         "entities": -1, "details": 0, "terminology": 0, "tone": 1}
    #   }
    # }

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
        score = item[metric]
        # if item["coherence"] < 2:
        #     continue
        # if score == -1:
        #     continue
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

def plot_score_proportions(data_dicts, metric, output_image=None):
    plt.figure(figsize=(12, 6))

    # Process each comparison type
    for label, data_dict in data_dicts.items():
        # Load and process data
        data_list = list(data_dict.values())
        scores = process_scores(data_list, metric)

        proportions = calculate_score_proportions(scores)

        # Get unique scores for x-axis
        unique_scores = sorted(set(scores))

        # Plot as lines
        # Plot stacked bars for each score threshold
        # Create a base color for this label using a consistent mapping
        label_index = list(data_dicts.keys()).index(label)
        base_color = plt.cm.Pastel1(label_index / len(data_dicts))

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

def check_references_match(data_dicts):
    references = {}
    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            references[int(index)] = item["reference"]
        break

    for data_type, data_dict in data_dicts.items():
        for index, item in data_dict.items():
            if item["reference"] != references[int(index)]:
                print(f"{data_type} {index} {item['reference']} != {references[int(index)]}")

def get_correlation_matrix_between_metrics(data_dict, metrics):
    """ correlation between metrics for each item in data_dict """
    # Get all scores for each metric
    all_scores = defaultdict(list)
    scores = get_scores(list(data_dict.values()))

    # convert scores[item_index]["metric"] to scores[metric][item_index]
    for metric in metrics:
        for score in scores:
            all_scores[metric].append(score[metric])

    # Get correlation matrix
    corr_matrix = np.corrcoef(list(all_scores.values()))
    return corr_matrix

def plot_correlation_matrix_between_metrics(data_dict, metrics):
    corr_matrix = get_correlation_matrix_between_metrics(data_dict, metrics)
    corr_matrix = pd.DataFrame(corr_matrix, index=metrics, columns=metrics)
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.yticks(np.arange(len(metrics))+0.5, metrics, rotation=0)
    plt.title("Correlation Matrix of Metrics")
    plt.show()
    return corr_matrix

def get_correlation_matrix_between_types(data_dicts, metric):
    """ correlation between score for the same metric between items in data_dict
    """
    if not np.all([len(v) == len(list(data_dicts.values())[0]) for v in data_dicts.values()]):
        data_dicts = get_data_dict_with_common_indices(data_dicts)
    check_references_match(data_dicts)

    all_scores = defaultdict(list)
    for data_type, data_dict in data_dicts.items():
        scores = get_scores(list(data_dict.values()))
        for score in scores:
            all_scores[data_type].append(score[metric])

    # Get correlation matrix
    corr_matrix = np.corrcoef(list(all_scores.values()))
    return corr_matrix

def plot_correlation_matrix_between_types(data_dicts, metric):
    corr_matrix = get_correlation_matrix_between_types(data_dicts, metric)
    corr_matrix = pd.DataFrame(corr_matrix, index=data_dicts.keys(), columns=data_dicts.keys())
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title(metric)
    plt.show()
    return corr_matrix




def get_data_dict_with_common_indices(data_dicts, compare_len=True):

    if compare_len:
        # check number of items in each data_dict before
        len_before = [len(v) for v in data_dicts.values()]

        # save average length of reference and comparison before
        mean_lens = {}
        get_len = lambda x, key: int(np.mean([len(v[key]) for v in x.values()]))
        for type, data_dict in data_dicts.items():
            mean_lens[type] = (get_len(data_dict, "reference"), get_len(data_dict, "comparison"))


    data_dicts = data_dicts.copy()
    indices = {}
    for k, v in data_dicts.items():
        indices[k] = [item["index"] for item in v.values()]

    common_indices = None
    for k, v in indices.items():
        if common_indices is None:
            common_indices = set(v)
        common_indices.intersection_update(v)

    for ref_type, data_dict in data_dicts.items():
        data_dicts[ref_type] = {str(k): v for k, v in data_dict.items() if int(k) in common_indices}


    if compare_len:
        len_after = [len(v) for v in data_dicts.values()]
        for idx, (type, data_dict) in enumerate(data_dicts.items()):
            mean_len_curr = (get_len(data_dict, "reference"), get_len(data_dict, "comparison"))
            print(f"{type:12}"
                  f" Ref comp: {mean_lens[ref_type]} -> {mean_len_curr}"
                  f" Len: {len_before[idx]:5} -> {len_after[idx]:5}")
    return data_dicts

if __name__ == "__main__":

    # Manually list the files.
    with open("processed_rubrics/all_data_dicts.json", "r") as f:
        data_dicts = json.load(f)

    # del data_dicts["cheat-10"]
    get_len = lambda x, key: int(np.mean([len(v[key]) for v in x.values()]))

    a = data_dicts["regenerated"]
    b = data_dicts["cheat-10"]
    indices_a = [item["index"] for item in a.values()]
    indices_b = [item["index"] for item in b.values()]
    short_indices = set(indices_a) - set(indices_b)
    print("Num short indices:", len(short_indices))
    for idx in sorted(short_indices)[:10]:
        print(idx, {"short": a[str(idx)]["reference"]})

    data_dicts = get_data_dict_with_common_indices(data_dicts)


    metrics = ["complexity", "coherence", "structure", "subject", "entities", "details", "terminology", "tone"]
    # metrics = ["coherence", "structure", "subject", "entities", "details"]

    for metric in metrics:
        plot_score_proportions(data_dicts, metric, f"{metric}.png")

    # for data_type, data_dict in data_dicts.items():
    #     plot_correlation_matrix(data_dict, metrics)

    # for metric in metrics:
    #     plot_correlation_matrix_between_types(data_dicts, metric)

# %%
