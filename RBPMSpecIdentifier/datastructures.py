import multiprocessing
import os

import pandas as pd
import numpy as np
from typing import Callable
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from RBPMSpecIdentifier.stats import fit_ecdf, get_permanova_results
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests
from statsmodels.distributions.empirical_distribution import ECDF




class RBPMSpecData:
    methods = {
        "Jensen-Shannon-Distance": "jensenshannon",
        "KL-Divergence": "symmetric-kl-divergence",
        "Euclidean-Distance": "euclidean"
    }
    def __init__(self, df: pd.DataFrame, design: pd.DataFrame, logbase: int = None):
        self.df = df
        self.logbase = logbase
        self.design = design
        self.array = None
        self.internal_design_matrix = None
        self.current_kernel_size = None
        self.norm_array = None
        self.distances = None
        self.anosim_ecdf = None
        self.permanova_ecdf = None
        self.pvalues = None
        self._data_rows = None
        self.current_eps = None
        self.indices_true = None
        self.indices_false = None
        self.internal_index = pd.DataFrame()
        self.permutation_sufficient_samples = False
        self._check_design()
        self._check_dataframe()




        self.calculated_score_names = ["RBPMSScore", "ANOSIM R", "global ANOSIM adj p-Value", "PERMANOVA F", "global PERMANOVA adj p-Value", "Mean Distance", "shift direction", "RNAse False peak pos", "RNAse True peak pos",  "Permanova p-value", "Permanova adj-p-value"]
        self.id_columns = ["RBPMSpecID", "id"]
        self.extra_columns = None

        self._set_design_and_array()




    def __getitem__(self, item):
        index = self.df.index.get_loc(item)
        return self.norm_array[index], self.distances[index]

    def _check_dataframe(self):
        if not pd.api.types.is_string_dtype(self.df.index.dtype):
            raise ValueError("The dataframe must have a string type index")

        if not set(self.design["Name"]).issubset(set(self.df.columns)):
            raise ValueError("Not all Names in the designs Name column are columns in the count df")

    def _check_design(self):
        for col in ["Fraction", "RNAse", "Replicate", "Name"]:
            if not col in self.design.columns:
                raise IndexError(f"{col} must be a column in the design dataframe\n")

    def _set_design_and_array(self):
        design_matrix = self.design.sort_values(by="Fraction")
        tmp = design_matrix.groupby(["RNAse", "Replicate"])["Name"].apply(list).reset_index()
        self.permutation_sufficient_samples = np.all(tmp.groupby("RNAse", group_keys=True)["Replicate"].count() >= 5)
        l = []
        rnames = []
        for idx, row in tmp.iterrows():
            sub_df = self.df[row["Name"]].to_numpy()
            rnames += row["Name"]
            l.append(sub_df)
        self.df["id"] = self.df.index
        self.df["RBPMSpecID"] = self.df.index
        self._data_rows = rnames
        self.extra_columns = [col for col in self.df.columns if col not in self._data_rows + self.id_columns]
        array = np.stack(l, axis=1)
        if self.logbase is not None:
            array = np.power(self.logbase, array)
            mask = np.isnan(array)
            array[mask] = 0
        self.array = array
        self.internal_design_matrix = tmp
        indices = self.internal_design_matrix.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        self.indices_false = indices[False]
        self.indices_true = indices[True]



    @property
    def extra_df(self):
        if self._data_rows is None:
            return None
        return self.df.iloc[:, ~np.isin(self.df.columns, self._data_rows)]

    @staticmethod
    def _normalize_rows(array, eps: float = 0):
        if eps:
            array += eps
        array = array / np.sum(array, axis=-1, keepdims=True)
        return array

    def normalize_array_with_kernel(self, kernel_size: int = 0, eps: float = 0):
        array = self.array
        self.current_kernel_size = kernel_size
        self.current_eps = eps

        if kernel_size:
            if not kernel_size % 2:
                raise ValueError(f"Kernel size must be odd")
            kernel = np.ones(kernel_size) / kernel_size
            array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=-1, arr=array)

        self.norm_array = self._normalize_rows(array, eps=eps)

    def calc_distances(self, method: str):
        if method == "jensenshannon":
            self.distances = self._jensenshannondistance(self.norm_array)
        elif method == "symmetric-kl-divergence":
            if self.current_eps is None or self.current_eps <= 0:
                raise ValueError(
                    "Cannot calculate KL-Divergence for Counts with 0 entries. "
                    "Need to set epsilon which is added to the raw Protein counts"
                )
            self.distances = self._symmetric_kl_divergence(self.norm_array)
        elif method == "euclidean":
            self.distances = self._jensenshannondistance(self.norm_array)
        else:
            raise ValueError(f"methhod: {method} is not supported")

    def _unset_scores_and_pvalues(self):
        for name in self.calculated_score_names:
            if name in self.df:
                self.df = self.df.drop(name, axis=1)


    def normalize_and_get_distances(self, method: str, kernel: int = 0, eps: float = 0):
        self.normalize_array_with_kernel(kernel, eps)
        self.calc_distances(method)
        self._unset_scores_and_pvalues()

    def determine_peaks(self):
        indices = self.internal_design_matrix.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        rnase_false = self.norm_array[:, indices[False]].mean(axis=-2)
        rnase_true = self.norm_array[:, indices[True]].mean(axis=-2)
        mid = 0.5 * (rnase_true + rnase_false)
        rel1 = rel_entr(rnase_false, mid)
        idx = np.arange(0, rel1.shape[0])
        r1 = np.argmax(rel1, axis=-1)
        #rel1 = rel1[idx, r1]
        r1 = r1 + int(np.ceil(self.current_kernel_size / 2))
        self.df["RNAse False peak pos"] = r1

        rel2 = rel_entr(rnase_true, mid)
        r2 = np.argmax(rel2, axis=-1)
        kernel = np.ones(self.current_kernel_size)
        rel2 = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=-1, arr=rel2)
        #rel2 = rel2[idx, r2]
        rel2 = jensenshannon(rnase_true, rnase_false, axis=-1, base=2)

        r2 = r2 + int(np.ceil(self.current_kernel_size / 2))
        self.df["RNAse True peak pos"] = r2
        self.df["Mean Distance"] = rel2
        side = r1 - r2
        side[side < 0] = -1
        side[side > 0] = 1
        shift_strings = np.empty(side.shape, dtype='U10')
        shift_strings = np.where(side == 0, "no shift", shift_strings)
        shift_strings = np.where(side == -1, "right", shift_strings)
        shift_strings = np.where(side == 1, "left", shift_strings)
        self.df["shift direction"] = shift_strings



    @staticmethod
    def _jensenshannondistance(array) -> np.ndarray:
        return jensenshannon(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1), base=2, axis=-2)

    @staticmethod
    def _symmetric_kl_divergence(array):
        r1 = rel_entr(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1)).sum(axis=-2)
        r2 = rel_entr(array[:, :, :, None].transpose(0, 3, 2, 1), array[:, :, :, None]).sum(axis=-2)
        return r1 + r2

    @staticmethod
    def euclidean_distance(array):
        return np.linalg.norm(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1), axis=-2)

    @staticmethod
    def calc_observation_pvalue(ecdf, distances, internal_design):
        indices = internal_design.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        mg1, mg2 = np.meshgrid(indices[True], indices[False])
        mg = np.stack((mg1, mg2))
        og_distances = distances.flat[np.ravel_multi_index(mg, distances.shape)]
        og_distances = og_distances.flatten()
        return 1 - ecdf(og_distances.mean())


    def _get_outer_group_distances(self, indices_false, indices_true):
        n_genes = self.distances.shape[0]
        mg1, mg2 = np.meshgrid(indices_true, indices_false)
        e = np.ones((n_genes, 3, 3))
        e = e * np.arange(0, n_genes)[:, None, None]
        e = e[np.newaxis, :]
        e = e.astype(int)
        mg = np.stack((mg1, mg2))

        mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)

        idx = np.concatenate((e, mg))
        distances = self.distances
        distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
        distances = distances.reshape((n_genes, len(indices_true) * len(indices_false)))
        return distances

    def _get_innergroup_distances(self,  indices_false, indices_true):
        distances = self.distances
        indices = [indices_false, indices_true]
        inner_distances = []
        for eidx, (idx) in enumerate(indices):
            n_genes = distances.shape[0]
            mg1, mg2 = np.meshgrid(idx, idx)
            e = np.ones((n_genes, 3, 3))
            e = e * np.arange(0, n_genes)[:, None, None]
            e = e[np.newaxis, :]
            e = e.astype(int)
            mg = np.stack((mg1, mg2))
            mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)
            idx = np.concatenate((e, mg))
            ig_distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
            iidx = np.triu_indices(n=ig_distances.shape[1], m=ig_distances.shape[2], k=1)
            ig_distances = ig_distances[:, iidx[0], iidx[1]]
            inner_distances.append(ig_distances)
        return np.concatenate(inner_distances, axis=-1)


    def calc_all_scores(self):
        self.calc_all_anosim_value()
        self.determine_peaks()

    def calc_anosim(self, indices_false, indices_true):
        outer_group_distances = self._get_outer_group_distances(indices_false, indices_true)
        inner_group_distances = self._get_innergroup_distances(indices_false, indices_true)
        stat_distances = np.concatenate((outer_group_distances, inner_group_distances), axis=-1)
        mask = np.isnan(stat_distances).any(axis=-1)
        ranks = stat_distances.argsort(axis=-1).argsort(axis=-1)
        rb = np.mean(ranks[:, 0:outer_group_distances.shape[-1]], axis=-1)
        rw = np.mean(ranks[:, outer_group_distances.shape[-1]:], axis=-1)
        r = (rb - rw) / (ranks.shape[-1] / 2)
        r[mask] = np.nan
        return r

    def calc_permanova_f(self, indices_false, indices_true):
        assert len(indices_true) == len(indices_false), "PERMANOVA performs poorly for unbalanced study design"
        outer_group_distances = self._get_outer_group_distances(indices_false, indices_true)
        inner_group_distances = self._get_innergroup_distances(indices_false, indices_true)
        bn = len(indices_true) + len(indices_false)
        n = len(indices_true)
        sst = np.sum(
            np.square(
                np.concatenate(
                    (outer_group_distances, inner_group_distances),
                    axis=-1
                )
            ), axis=-1
        ) / bn
        ssw = np.sum(np.square(inner_group_distances), axis=-1) / n
        ssa = sst - ssw
        f = (ssa) / (ssw / (bn-2))
        return f

    def calc_all_permanova_f(self):
        f = self.calc_permanova_f(self.indices_false, self.indices_true)
        self.df["PERMANOVA F"] = f

    def calc_all_anosim_value(self):
        r = self.calc_anosim(self.indices_false, self.indices_true)
        self.df["ANOSIM R"] = r

    def _calc_global_anosim_distribution(self, nr_permutations: int, threads: int, seed: int = 0):
        np.random.seed(seed)
        _split_point = len(self.indices_false)
        indices = np.concatenate((self.indices_false, self.indices_true))
        calls = []
        for _ in range(nr_permutations):
            shuffled = np.random.permutation(indices)
            calls.append((shuffled[:_split_point], shuffled[_split_point:]))

        with multiprocessing.Pool(threads) as pool:
            result = pool.starmap(self.calc_anosim, calls)
        result = np.concatenate(result)
        result = result[~np.isnan(result)] + 1e-15
        self.anosim_ecdf = ECDF(result)

    def _calc_global_permanova_distribution(self, nr_permutations: int, threads: int, seed: int = 0):
        np.random.seed(seed)
        _split_point = len(self.indices_false)
        indices = np.concatenate((self.indices_false, self.indices_true))
        calls = []
        for _ in range(nr_permutations):
            shuffled = np.random.permutation(indices)
            calls.append((shuffled[:_split_point], shuffled[_split_point:]))

        with multiprocessing.Pool(threads) as pool:
            result = pool.starmap(self.calc_permanova_f, calls)
        result = np.concatenate(result)
        result = result[~np.isnan(result)] + 1e-15
        self.permanova_ecdf = ECDF(result)

    def calc_global_anosim_p_value(self, permutations: int, threads: int, seed: int = 0, distance_cutoff: float = None):
        if "ANOSIM R" not in self.df.columns:
            self.calc_all_anosim_value()
        self._calc_global_anosim_distribution(permutations, threads, seed)
        p_values = 1 - self.anosim_ecdf(self.df["ANOSIM R"])
        mask = self.df["ANOSIM R"].isna()
        if distance_cutoff is not None:
            if "Mean Distance" not in self.df.columns:
                raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
            mask[self.df["Mean Distance"] < distance_cutoff] = True
        p_values[mask] = np.nan
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df["global ANOSIM adj p-Value"] = p_values

    def calc_global_permanova_p_value(self, permutations: int, threads: int, seed: int = 0, distance_cutoff: float = None):
        if "PERMANOVA F" not in self.df.columns:
            self.calc_all_permanova_f()
        self._calc_global_permanova_distribution(permutations, threads, seed)
        p_values = 1 - self.permanova_ecdf(self.df["PERMANOVA F"])
        mask = self.df["PERMANOVA F"].isna()
        if distance_cutoff is not None:
            if "Mean Distance" not in self.df.columns:
                raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
            mask[self.df["Mean Distance"] < distance_cutoff] = True
        p_values[mask] = np.nan
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df["global PERMANOVA adj p-Value"] = p_values

    def export_csv(self, file: str,  sep: str = ","):
        self.extra_df.to_csv(file, sep=sep)

    def calc_all_permanova(self, permutations, threads):
        calls = []
        for idx in range(self.distances.shape[0]):
            d = self.distances[idx]
            if ~np.any(np.isnan(d)):

                calls.append((d, self.internal_design_matrix, permutations, self.df.index[idx]))
        with Pool(threads) as pool:
            data = pool.starmap(get_permanova_results, calls)
        permanova_results = pd.concat(data, axis=1).T.set_index("gene_id")
        _, permanova_results["adj-p-value"], _, _ = multipletests(permanova_results["p-value"], method="fdr_bh")
        self.df["Permanova p-value"] = permanova_results["p-value"]
        self.df["Permanova adj-p-value"] = permanova_results["adj-p-value"]


def _analysis_executable_wrapper(args):
    design = pd.read_csv(args.design_matrix, sep=args.sep)
    df = pd.read_csv(args.input, sep=args.sep)
    rbpmspec = RBPMSpecData(df, design, args.logbase)
    kernel_size = args.kernel_size if args.kernel_size > 0 else 0
    rbpmspec.normalize_and_get_distances(args.distance_method, kernel_size, args.eps)
    rbpmspec.calc_all_scores()
    rbpmspec.calc_all_permanova(999, 5)





if __name__ == '__main__':
    df = pd.read_csv("../testData/testFile.tsv", sep="\t", index_col=0)
    #sdf = df[[col for col in df.columns if "LFQ" in col]]
    sdf = df
    sdf.index = sdf.index.astype(str)
    design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
    rbpmspec = RBPMSpecData(sdf, design, logbase=2)
    rbpmspec.normalize_and_get_distances("jensenshannon", 3)
    rbpmspec.calc_global_permanova_p_value(1000, threads=5)
    rbpmspec.calc_all_permanova(1000, threads=5)
    rbpmspec.export_csv(file="foofile.tsv", sep="\t")
    import plotly.graph_objs as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rbpmspec.permanova_ecdf.x, y=rbpmspec.permanova_ecdf.y))
    fig.show()

