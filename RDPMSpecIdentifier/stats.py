from skbio.stats.distance import permanova
from skbio.stats.distance import DistanceMatrix
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from scipy.special import rel_entr
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
from multiprocessing import Pool
from itertools import product
from scipy.stats import mannwhitneyu
from statsmodels.distributions.empirical_distribution import ECDF



def normalize_rows(array, alpha: float = 0):
    if alpha:
        array += alpha
    array = array / np.sum(array, axis=-1, keepdims=True)
    return array


def pca(array: np.ndarray):
    array = array.transpose((1, 0, 2))
    array = array.reshape(-1, array.shape[1] * array.shape[2])
    array = array[ :, ~np.isnan(array).any(axis=0)]
    pca = PCA()
    components = pca.fit_transform(array)
    return components








def generate_matrix(df, design_matrix):
    design_matrix = design_matrix.sort_values(by="fraction")
    tmp = design_matrix.groupby(["RNAse", "replicate"])["name"].apply(list).reset_index()
    l = []
    for idx, row in tmp.iterrows():
        sdf = df[row["name"]].to_numpy()
        l.append(sdf)
    array = np.stack(l, axis=1)
    return array, tmp




def get_permanova_results(distances, design_matrix, permutations, gene_id):
    entry = DistanceMatrix(distances)
    res = permanova(entry, grouping=design_matrix["RNAse"], permutations=permutations)
    res["gene_id"] = gene_id
    if np.isnan(res["test statistic"]):
        res["p-value"] = np.nan
    return res


def mann_whitney_vs_background(distances, design_matrix, group, levels, ig_distances):
    indices = design_matrix.groupby(group, group_keys=True).apply(lambda x: list(x.index))
    mg1, mg2 = np.meshgrid(indices[levels[0]], indices[levels[1]])
    mg = np.stack((mg1, mg2))
    og_distances = distances.flat[np.ravel_multi_index(mg, distances.shape)]
    og_distances = og_distances.flatten()
    statistic, pvalue = mannwhitneyu(og_distances, ig_distances)
    return pvalue, og_distances



def calc_innergroup_background(distances, design_matrix, groups):
    distances = distances[~np.isnan(distances).any(axis=(1, 2)), :, :]
    idx = distances == 0
    inner_distances = []
    indices = design_matrix.groupby(groups, group_keys=True).apply(lambda x: list(x.index))
    for eidx, (name, idx) in enumerate(indices.items()):
        n_genes = distances.shape[0]
        mg1, mg2 = np.meshgrid(idx, idx)
        e = np.ones((n_genes, 3, 3))
        e = e * np.arange(0, n_genes)[:, None, None]
        e = e[np.newaxis, :]
        e = e.astype(int)
        mg = np.stack((mg1, mg2))

        mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)

        idx = np.concatenate((e, mg))
        # mg1, mg2 = np.meshgrid(idx, idx)
        # mg = np.stack((mg2[np.triu_indices(len(idx), k=1)], mg1[np.triu_indices(len(idx), k=1)]))
        # rep_idx = np.repeat(mg, distances.shape[0], axis=1)
        # rep_idx2 = np.tile(np.arange(distances.shape[0]), mg.shape[1])
        #
        # idx = np.concatenate((rep_idx2[None, :], rep_idx), axis=0)

        ig_distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
        iidx = np.triu_indices(n=ig_distances.shape[1], m=ig_distances.shape[2], k=1)
        ig_distances = ig_distances[:, iidx[0], iidx[1]].mean(axis=-1)
        inner_distances.append(ig_distances)
    inner_distances = np.concatenate(inner_distances)
    inner_distances = inner_distances[~np.isnan(inner_distances)]
    inner_distances = inner_distances[~(inner_distances == 0)]
    return inner_distances


def fit_ecdf(distances, design_matrix, groups):
    inner_distances = calc_innergroup_background(distances, design_matrix, groups)
    return ECDF(inner_distances)

