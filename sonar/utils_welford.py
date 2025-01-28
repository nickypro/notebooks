import pickle
import os
import gc
from tqdm import tqdm
import torch
from welford_torch import Welford
from utils_load_data import load_res_data, load_embeds

def load_or_compute_welford_stats(groups_to_load):
    """Load or compute Welford statistics for normalization"""
    os.makedirs('./welford_data', exist_ok=True)
    welford_file = f'./welford_data/welford_stats_10_{groups_to_load}.pkl'

    try:
        with open(welford_file, 'rb') as f:
            print("Loading existing welford data")
            welford_stats = pickle.load(f)
            return welford_stats['welford_emb'], welford_stats['welford_res']

    except FileNotFoundError:
        welford_emb = Welford()
        welford_res = Welford()

        for i in tqdm(range(10)):
            res_data = load_res_data(i, groups_to_load=groups_to_load).cuda()
            embeds = load_embeds(i).cuda()

            welford_res.add_all(res_data)
            welford_emb.add_all(embeds)
            del res_data, embeds
            gc.collect()
            torch.cuda.empty_cache()

        # Save Welford statistics for first 10 files using pickle
        with open(welford_file, 'wb') as f:
            pickle.dump({
                'welford_emb': welford_emb,
                'welford_res': welford_res
            }, f)

        return welford_emb, welford_res

def normalize_res(res_data, welford_res):
    """Normalize residual data using precomputed mean and variance"""
    return (res_data - welford_res.mean) / torch.sqrt(welford_res.var_s + 1e-8)

def normalize_emb(emb_data, welford_emb):
    """Normalize embedding data using precomputed mean and variance"""
    return (emb_data - welford_emb.mean) / torch.sqrt(welford_emb.var_s + 1e-8)

def restore_emb(normed_emb_data, welford_emb):
    """Restore normalized embedding data to original scale using precomputed mean and variance"""
    return normed_emb_data * torch.sqrt(welford_emb.var_s + 1e-8) + welford_emb.mean
