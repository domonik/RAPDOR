import multiprocessing

import pandas as pd
import numpy as np
from typing import Callable
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from RBPMSpecIdentifier.stats import fit_ecdf, get_permanova_results
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests




class RBPMSpecData:
    methods = {
        "Jensen-Shannon-Distance": "jensenshannon",
        "KL-Divergence": "symmetric-kl-divergence"
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
        self.ecdf = None
        self.pvalues = None
        self._data_rows = None
        self.current_eps = None
        self.internal_index = pd.DataFrame()
        self.permanova_sufficient_samples = False
        self._check_design()
        self._check_dataframe()




        self.calculated_score_names = ["RBPMSScore", "ANOSIM R", "Permanova p-value", "Permanova adj-p-value"]
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
        self.permanova_sufficient_samples = np.all(tmp.groupby("RNAse", group_keys=True)["Replicate"].count() >= 5)
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
        else:
            raise ValueError(f"mehthod: {method} is not supported")

    def _unset_scores_and_pvalues(self):
        for name in self.calculated_score_names:
            if name in self.df:
                self.df = self.df.drop(name, axis=1)


    def normalize_and_get_distances(self, method: str, kernel: int = 0, eps: float = 0):
        self.normalize_array_with_kernel(kernel, eps)
        self.calc_distances(method)
        self._unset_scores_and_pvalues()



    @staticmethod
    def _jensenshannondistance(array) -> np.ndarray:
        return jensenshannon(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1), axis=-2)

    @staticmethod
    def _symmetric_kl_divergence(array):
        r1 = rel_entr(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1)).sum(axis=-2)
        r2 = rel_entr(array[:, :, :, None].transpose(0, 3, 2, 1), array[:, :, :, None]).sum(axis=-2)
        return r1 + r2

    def fit_innergroup_ecdf(self):
        ecdf = fit_ecdf(self.distances, self.internal_design_matrix, "RNAse")
        self.ecdf = ecdf

    @staticmethod
    def calc_observation_pvalue(ecdf, distances, internal_design):
        indices = internal_design.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        mg1, mg2 = np.meshgrid(indices[True], indices[False])
        mg = np.stack((mg1, mg2))
        og_distances = distances.flat[np.ravel_multi_index(mg, distances.shape)]
        og_distances = og_distances.flatten()
        return 1 - ecdf(og_distances.mean())


    def _get_outer_group_distances(self):
        n_genes = self.distances.shape[0]
        indices = self.internal_design_matrix.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        mg1, mg2 = np.meshgrid(indices[True], indices[False])
        e = np.ones((n_genes, 3, 3))
        e = e * np.arange(0, n_genes)[:, None, None]
        e = e[np.newaxis, :]
        e = e.astype(int)
        mg = np.stack((mg1, mg2))

        mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)

        idx = np.concatenate((e, mg))
        mask = np.any(np.isnan(self.distances), axis=(-1, -2))
        distances = self.distances
        distances[mask] = np.nan
        distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
        distances = distances.reshape((n_genes, len(indices[True]), len(indices[False])))
        indices1, indices2 = np.triu_indices(n=len(indices[True]), m=len(indices[False]))
        distances = distances[:, indices1, indices2]
        return distances

    def _get_innergroup_distances(self):
        distances = self.distances
        design_matrix = self.internal_design_matrix
        inner_distances = []
        indices = design_matrix.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
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
            ig_distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
            iidx = np.triu_indices(n=ig_distances.shape[1], m=ig_distances.shape[2], k=1)
            ig_distances = ig_distances[:, iidx[0], iidx[1]]
            inner_distances.append(ig_distances)
        return np.concatenate(inner_distances, axis=-1)

    def calc_rbp_scores(self):
        self.fit_innergroup_ecdf()
        n_genes = self.distances.shape[0]
        indices = self.internal_design_matrix.groupby("RNAse", group_keys=True).apply(lambda x: list(x.index))
        mg1, mg2 = np.meshgrid(indices[True], indices[False])
        e = np.ones((n_genes, 3, 3))
        e = e * np.arange(0, n_genes)[:, None, None]
        e = e[np.newaxis, :]
        e = e.astype(int)
        mg = np.stack((mg1, mg2))

        mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)

        idx = np.concatenate((e, mg))
        mask = np.any(np.isnan(self.distances), axis=(-1, -2))
        distances = self.distances
        distances[mask] = np.nan
        distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
        distances = distances.reshape((n_genes, -1))
        distances = np.mean(distances, axis=1)
        ecdf = self.ecdf(distances)
        ecdf[mask] = np.nan
        self.pvalues = ecdf
        self.df["RBPMSScore"] = self.pvalues
        self.calc_all_anosim_value()

    def calc_all_scores(self):
        self.calc_all_anosim_value()
        self.calc_rbp_scores()

    def calc_all_anosim_value(self):
        outer_group_distances = self._get_outer_group_distances()
        inner_group_distances = self._get_innergroup_distances()
        stat_distances = np.concatenate((outer_group_distances, inner_group_distances), axis=-1)
        ranks = stat_distances.argsort(axis=-1).argsort(axis=-1)
        rb = np.mean(ranks[:, 0:outer_group_distances.shape[-1]], axis=-1)
        rw = np.mean(ranks[:, outer_group_distances.shape[-1]:], axis=-1)
        r = (rb - rw) / (ranks.shape[-1] / 2)
        self.df["ANOSIM R"] = r



    def export_csv(self, file: str,  sep: str = ","):
        self.df.to_csv(file, sep=sep)

    def calc_all_permanova(self, permutations, num_threads):
        calls = []
        for idx in range(self.distances.shape[0]):
            d = self.distances[idx]
            if ~np.any(np.isnan(d)):

                calls.append((d, self.internal_design_matrix, permutations, self.df.index[idx]))
        with Pool(num_threads) as pool:
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
    sdf = sdf.fillna(0)
    sdf.index = sdf.index.astype(str)
    design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
    rbpmspec = RBPMSpecData(sdf, design, logbase=2)
    rbpmspec.normalize_and_get_distances("jensenshannon", 3)
    rbpmspec.calc_all_anosim_value()
