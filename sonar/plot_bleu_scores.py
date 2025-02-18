# %%
from utils_plot import load_rubric_results
from tqdm import tqdm
import sacrebleu
import numpy as np

data_dicts = load_rubric_results(indices_intersection=False)

# dict_keys(['mlp', 'linear', 'continued', 'baseline', 'cheat-1', 'cheat-5', 'cheat-10', 'regenerated', 'auto-decoded'])


references = [['the quick brown fox jumps']]  # Reference text
candidates = ['the fast brown fox leaps']     # Generated text

score = sacrebleu.sentence_bleu(candidates[0], references[0])
print(f"Standardized BLEU: {score.score:.2f}")

bleu_scores = {}

for data_type, data_dict in data_dicts.items():
    bleu_scores[data_type] = []
    for index, datum in tqdm(data_dict.items()):
        ref_text = datum['reference']
        gen_text = datum['comparison']
        score = sacrebleu.sentence_bleu(gen_text, [ref_text])
        bleu_scores[data_type].append(score.score)
    print(f"{data_type}: {np.mean(bleu_scores[data_type]):.2f} ± {np.std(bleu_scores[data_type]):.2f}")



# %%
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(
    target="The capital of France is Paris",
    prediction="Paris serves as France's capital"
)
print(f"Precision: {scores['rouge1'].precision:.2f}, Recall: {scores['rouge1'].recall:.2f}")
print(f"Precision: {scores['rouge2'].precision:.2f}, Recall: {scores['rouge2'].recall:.2f}")
print(f"Precision: {scores['rougeL'].precision:.2f}, Recall: {scores['rougeL'].recall:.2f}")

rouge_scores = {}

for data_type, data_dict in data_dicts.items():
    rouge_scores[data_type] = {"rouge1": [], "rouge2": [], "rougeL": []}
    for index, datum in tqdm(data_dict.items()):
        ref_text = datum['reference']
        gen_text = datum['comparison']
        scores = scorer.score(ref_text, gen_text)
        rouge_scores[data_type]["rouge1"].append(scores['rouge1'].precision)
        rouge_scores[data_type]["rouge2"].append(scores['rouge2'].precision)
        rouge_scores[data_type]["rougeL"].append(scores['rougeL'].precision)
    print(f"{data_type}: {np.mean(rouge_scores[data_type]['rouge1']):.2f} ± {np.std(rouge_scores[data_type]['rouge1']):.2f}, {np.mean(rouge_scores[data_type]['rouge2']):.2f} ± {np.std(rouge_scores[data_type]['rouge2']):.2f}, {np.mean(rouge_scores[data_type]['rougeL']):.2f} ± {np.std(rouge_scores[data_type]['rougeL']):.2f}")

# %%

for data_type, bleu_score in bleu_scores.items():
    print(f"{data_type}: {np.mean(bleu_score):.2f} ± {np.std(bleu_score):.2f}")

for data_type, rouge_score in rouge_scores.items():
    # print(f"{data_type}: {np.mean(rouge_score['rouge1']):.2f} ± {np.std(rouge_score['rouge1']):.2f}, {np.mean(rouge_score['rouge2']):.2f} ± {np.std(rouge_score['rouge2']):.2f}, {np.mean(rouge_score['rougeL']):.2f} ± {np.std(rouge_score['rougeL']):.2f}")

# %%


