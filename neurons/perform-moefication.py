import numpy as np
from sklearn.preprocessing import normalize as sk_normalize
from k_means_constrained import KMeansConstrained
from taker import Model
from tqdm import tqdm

def cluster_neurons(model: Model,
        layer: int,
        split_num: int=96,
        method="kmeans",
    ):
    # First, get variables for which components are used
    assert model.cfg.d_mlp % split_num == 0, \
        f"split_num {split_num} should evenly divide model's mlp width {model.cfg.d_mlp}"
    split_size = model.cfg.d_mlp // split_num

    # Collect the neurons we are clustering
    weights = model.layers[layer]["mlp.W_in"].detach().cpu()
    normed_weights = sk_normalize(weights)

    # Perform the clustering
    if method == "kmeans":
        kmeans = KMeansConstrained(
            n_clusters=split_num, size_min=split_size, size_max=split_size, random_state=0
        ).fit(normed_weights, None)
        labels = [x for x in kmeans.labels_]
        return labels, kmeans

    if method == "random":
         labels = np.array(list(range(model.cfg.d_mlp))) % split_num


def get_clustering(m, split_num=128):
    clusters = []

    for layer in tqdm(range(m.cfg.n_layers)):
        cluster_i, _ = cluster_neurons(m, layer, split_num=split_num)
        print(cluster_i)
        cluster_i += np.ones_like(cluster_i) * split_num * layer
        clusters.append( cluster_i )

    clusters_list = np.array( clusters )
    clusters_list.shape
    return clusters_list


#m_good = Model("NousResearch/Llama-2-7b-hf")
#m_good = Model("nickypro/llama-7b-hf-rand")
m_good = Model("nickypro/mistral-7b-rand")
cluster_list = get_clustering(m_good, 128)
np.save('mistral-7b-rand-moe.npy', np.array(cluster_list))
