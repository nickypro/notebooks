# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from utils_plot import load_rubric_results, get_data_dict_with_common_indices, run_references_match_check, get_scores

def get_correlation_matrix(values, method='spearman'):
    """
    Calculate correlation matrix for a list of values using specified method.

    Args:
        values: List of value arrays to correlate
        method: Correlation method to use - one of:
            'pearson': Pearson correlation coefficient
            'spearman': Spearman rank correlation
            'kendalltau': Kendall's tau correlation
            'cohen_kappa': Cohen's kappa coefficient
    """
    num_vars = len(values)
    corr_matrix = np.zeros((num_vars, num_vars))

    for i in range(num_vars):
        for j in range(num_vars):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                if method == 'pearson':
                    corr, _ = stats.pearsonr(values[i], values[j])
                elif method == 'spearman':
                    corr, _ = stats.spearmanr(values[i], values[j])
                elif method == 'kendalltau':
                    corr, _ = stats.kendalltau(values[i], values[j])
                elif method == 'cohen_kappa':
                    corr = cohen_kappa_score(values[i], values[j])
                else:
                    raise ValueError("method must be one of: 'pearson', 'spearman', 'kendalltau', 'cohen_kappa'")
                corr_matrix[i, j] = corr

    return corr_matrix
def get_correlation_matrix_between_metrics(data_dict: list[dict], metrics: list, method: str ='spearman'):
    """ correlation between metrics for each item in data_dict """
    # Get all scores for each metric
    all_scores = defaultdict(list)
    scores = get_scores(list(data_dict.values()))

    # convert scores[item_index]["metric"] to scores[metric][item_index]
    for metric in metrics:
        if metric == "length":
            continue
        for score in scores:
            all_scores[metric].append(score[metric])

    if "length" in metrics:
        for item in data_dict.values():
            all_scores["length"].append(len(item["reference"]))

    # Get correlation matrix
    corr_matrix = get_correlation_matrix(list(all_scores.values()), method=method)

    return corr_matrix

def plot_correlation_matrix_between_metrics(data_dict, metrics, type=None, method='spearman'):
    corr_matrix = get_correlation_matrix_between_metrics(data_dict, metrics, method=method)
    corr_matrix = pd.DataFrame(corr_matrix, index=metrics, columns=metrics)
    fig = plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt='.2f')
    plt.yticks(np.arange(len(metrics))+0.5, metrics, rotation=0)
    plt.xticks(np.arange(len(metrics))+0.5, metrics, rotation=30, ha='right')
    plt.title(f"Correlation Matrix of Metrics ({type})")
    return fig

def get_correlation_matrix_between_types(data_dicts, metric, method='spearman'):
    """ correlation between score for the same metric between items in data_dict
    """
    if not np.all([len(v) == len(list(data_dicts.values())[0]) for v in data_dicts.values()]):
        data_dicts = get_data_dict_with_common_indices(data_dicts)
    run_references_match_check(data_dicts)

    all_scores = defaultdict(list)
    for data_type, data_dict in data_dicts.items():
        if metric == "length":
            scores = [{"length": len(item["comparison"])} for item in data_dict.values()]
        else:
            scores = get_scores(list(data_dict.values()))
        for score in scores:
            all_scores[data_type].append(score[metric])

    # Get correlation matrix
    corr_matrix = get_correlation_matrix(list(all_scores.values()), method=method)
    return corr_matrix

def plot_correlation_matrix_between_types(data_dicts, metric, method='spearman'):
    corr_matrix = get_correlation_matrix_between_types(data_dicts, metric, method=method)
    corr_matrix = pd.DataFrame(corr_matrix, index=data_dicts.keys(), columns=data_dicts.keys())
    fig = plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", vmin=0, vmax=1, fmt='.2f')
    plt.title(metric)
    plt.xticks(np.arange(len(data_dicts.keys()))+0.5, data_dicts.keys(), rotation=30, ha='right')
    plt.yticks(np.arange(len(data_dicts.keys()))+0.5, data_dicts.keys(), rotation=0)
    return fig

#######################################

if __name__ == "__main__":
    data_dicts = load_rubric_results(
        file_path="processed_rubrics/all_data_dicts.json",
        indices_intersection=True,
        check_short_indices=False,
        check_references_match=False,
    )

    metrics = ["complexity", "coherence", "structure", "subject", "entities", "details", "terminology", "tone", "length"]

    for metric in metrics:
        fig = plot_correlation_matrix_between_types(data_dicts, metric, method='spearman')
        fig.savefig(f"figures/correlation_matrix_between_types_{metric}.png")

    for data_type, data_dict in data_dicts.items():
        fig = plot_correlation_matrix_between_metrics(data_dict, metrics, type=data_type, method='spearman')
        fig.savefig(f"figures/correlation_matrix_between_metrics_{data_type}.png")

# %%
