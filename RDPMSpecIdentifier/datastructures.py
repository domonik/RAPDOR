import multiprocessing
import os
from scipy.stats import ttest_ind
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
from typing import Callable
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
# from RDPMSpecIdentifier.stats import fit_ecdf, get_permanova_results
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN
from umap import UMAP
from dataclasses import dataclass
import json
from json import JSONEncoder
from pandas.testing import assert_frame_equal
from statsmodels.distributions.empirical_distribution import ECDF

DECIMALS = 15

@dataclass()
class RDPMState:
    distance_method: str = None
    kernel_size: int = None
    eps: float = None
    dimension_reduction: str = None
    cluster_kernel_distance: int = None
    permanova: str = None
    permanova_permutations: int = None
    permanova_cutoff: float = None
    scored: bool = False
    anosim_r: bool = None
    permanova_f: bool = None
    cluster_method: str = None
    cluster_args: dict = None

    def to_json(self):
        return self.__dict__

    def __eq__(self, other):
        if not isinstance(other, RDPMState):
            return False
        else:
            other_dict = other.__dict__
            for key, value in self.__dict__.items():
                if value != other_dict[key]:
                    return False
            return True


class RDPMSpecData:
    r""" The RDPMSpec Class storing results and containing functions for analysis

     Attributes:
        df (pd.Dataframe): the dataframe that stores intensities and additional columns per protein.
        logbase (int): the logbase if intensities in :attr:`df` are log transformed. Else None.
        design (pd.Dataframe): dataframe containing information about the intensity columns in :attr:`df`
        array: (np.ndarray): The non-normalized intensities from the :attr:`df` intensity columns.
        internal_design_matrix (pd.Dataframe): dataframe where fraction columns are stored as a list instead of
            seperate columns
        current_kernel_size (Union[None, int]): If an averaging kernel was applied during `array` normalization,
            this stores the kernel size. If there was no normalization yet or no kernel was used it is None.
        norm_array (Union[None, np.ndarray]): An array containing normalized values that add up to 1.
        distances (Union[None, np.ndarray]): An array of size `num_proteins x num_samples x num_samples` that stores the
            distance between samples. If no distance was calculated it is None.
        permutation_sufficient_samples (bool): Set to true if there are at least 5 samples per condition. Else False.
        score_columns (List[str]): list of strings that are used as column names for scores that can be calculated via
            this object.


     Examples:
        An instance of the RDPMSpecData class is obtained via the following code. Make sure your csv files
        are correctly fomatted as desribed in the :ref:`Data Prepatation<data-prep-tutorial>` Tutorial.

        >>> df = pd.read_csv("../testData/testFile.tsv", sep="\t", index_col=0)
        >>> df.index = df.index.astype(str)
        >>> design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
        >>> rdpmspec = RDPMSpecData(df, design, logbase=2)
    """
    methods = [
        "Jensen-Shannon-Distance",
        "KL-Divergence",
        "Euclidean-Distance",
    ]
    score_columns = [
        "Rank",
        "RDPMSScore",
        "ANOSIM R",
        "global ANOSIM adj p-Value",
        "local ANOSIM adj p-Value",
        "PERMANOVA F",
        "global PERMANOVA adj p-Value",
        "local PERMANOVA adj p-Value",
        "Mean Distance",
        "shift direction",
        "RNase False peak pos",
        "RNase True peak pos",
        "Permanova p-value",
        "Permanova adj-p-value",
        "CTRL Peak adj p-Value",
        "RNase Peak adj p-Value"
    ]

    _id_columns = ["RDPMSpecID", "id"]

    def __init__(self, df: pd.DataFrame, design: pd.DataFrame, logbase: int = None):
        self.state = RDPMState()
        self.df = df
        self.logbase = logbase
        self.design = design
        self.array = None
        self.internal_design_matrix = None
        self.norm_array = None
        self.distances = None
        self._anosim_distribution = None
        self._permanova_distribution = None
        self._data_rows = None
        self._current_eps = None
        self._indices_true = None
        self._indices_false = None
        self.cluster_features = None
        self.current_embedding = None
        self.permutation_sufficient_samples = False
        self._check_design()
        self._check_dataframe()
        self._set_design_and_array()

    def __eq__(self, other):
        if not isinstance(other, RDPMSpecData):
            return False
        else:
            other_dict = other.__dict__
            for key, value in self.__dict__.items():
                other_item = other_dict[key]
                if not isinstance(value, type(other_item)):
                    return False
                elif isinstance(value, pd.DataFrame):
                    try:
                        assert_frame_equal(value, other_item, check_dtype=False)
                    except AssertionError:
                        return False
                elif isinstance(value, np.ndarray):
                    if value.dtype.kind in ["U", "S"]:
                        if not np.all(value == other_item):
                            return False
                    else:
                        if not np.allclose(value, other_item, equal_nan=True):
                            return False
                else:
                    if value != other_item:
                        return False
            return True

    def __getitem__(self, item):
        """
        Args:
            item (str): The index of the protein which data should be returned

        Returns:  Tuple[np.ndarray, np.ndarray]
            The normalized array of the protein with the id in `item` and the sample distances

        Raises:
            ValueError: If either the array was not normalized or distances were not calculated yet.

        """
        if self.norm_array is None:
            raise ValueError("Array is not normalized yet. Normalize it first")
        if self.distances is None:
            raise ValueError("Sample distances not calculated yet. Calculate them first")
        index = self.df.index.get_loc(item)
        return self.norm_array[index], self.distances[index]

    def _check_dataframe(self):
        if not set(self.design["Name"]).issubset(set(self.df.columns)):
            raise ValueError("Not all Names in the designs Name column are columns in the count df")

    def _check_design(self):
        for col in ["Fraction", "RNase", "Replicate", "Name"]:
            if not col in self.design.columns:
                raise IndexError(f"{col} must be a column in the design dataframe\n")

    def _set_design_and_array(self):
        design_matrix = self.design.sort_values(by="Fraction")
        tmp = design_matrix.groupby(["RNase", "Replicate"])["Name"].apply(list).reset_index()
        self.df.index = np.arange(self.df.shape[0])
        self.df = self.df.round(decimals=DECIMALS)

        self.permutation_sufficient_samples = bool(np.all(tmp.groupby("RNase", group_keys=True)["Replicate"].count() >= 5))
        l = []
        rnames = []
        for idx, row in tmp.iterrows():
            sub_df = self.df[row["Name"]].to_numpy()
            rnames += row["Name"]
            l.append(sub_df)
        self.df["RDPMSpecID"] = self.df.iloc[:, 0]
        self.df["id"] = self.df.index
        self._data_rows = np.asarray(rnames)
        array = np.stack(l, axis=1)
        if self.logbase is not None:
            array = np.power(self.logbase, array)
            mask = np.isnan(array)
            array[mask] = 0
        self.array = array
        self.internal_design_matrix = tmp
        indices = self.internal_design_matrix.groupby("RNase", group_keys=True).apply(lambda x: list(x.index))
        self._indices_false = np.asarray(indices[False])
        self._indices_true = np.asarray(indices[True])

    @classmethod
    def from_files(cls, intensities: str, design: str, logbase: int = None, sep: str = ","):
        """Constructor to generate instance from files instead of pandas dataframes.

        Args:
            intensities (str): Path to the intensities File
            design (str): Path to the design file
            logbase (Union[None, int]): Logbase if intensities in the intensity file are log transformed
            sep (str): seperator used in the intensities and design files. Must be the same for both.

        Returns: RDPMSpecData

        """
        design = pd.read_csv(design, sep=sep)
        df = pd.read_csv(intensities, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        rdpmspec = RDPMSpecData(df, design, logbase)
        return rdpmspec

    @property
    def extra_df(self):
        """ Return a Dataframe Slice all columns from self.df that are not part of the intensity columns

        Returns: pd.Dataframe

        """
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
        """Normalizes the array and sets `norm_array` attribute.

        Args:
            kernel_size (int): Averaging kernel size. This kernel is applied to the fractions.
            eps (float): epsilon added to the intensities to overcome problems with zeros.

        """
        array = self.array

        if kernel_size:
            if not kernel_size % 2:
                raise ValueError(f"Kernel size must be odd")
            kernel = np.ones(kernel_size) / kernel_size
            array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=-1, arr=array)

        self.norm_array = self._normalize_rows(array, eps=eps)
        self.state.kernel_size = kernel_size
        self.state.eps = eps

    def calc_distances(self, method: str):
        """Calculates between sample distances.
                
        Args:
            method (str): One of the values from `methods`. The method used for sample distance calculation.

        Raises:
            ValueError: If the method string is not supported or symmetric-kl-divergence is used without adding an
                epsilon to the protein intensities

        """
        if method == "Jensen-Shannon-Distance":
            self.distances = self._jensenshannondistance(self.norm_array)

        elif method == "KL-Divergence":
            if self.state.eps is None or self.state.eps <= 0:
                raise ValueError(
                    "Cannot calculate KL-Divergence for Counts with 0 entries. "
                    "Need to set epsilon which is added to the raw Protein intensities"
                )
            self.distances = self._symmetric_kl_divergence(self.norm_array)
        elif method == "Euclidean-Distance":
            self.distances = self._euclidean_distance(self.norm_array)
        else:
            raise ValueError(f"methhod: {method} is not supported")
        self.state.distance_method = method

    def _unset_scores_and_pvalues(self):
        for name in self.score_columns:
            if name in self.df:
                self.df = self.df.drop(name, axis=1)
        self.remove_clusters()

    def normalize_and_get_distances(self, method: str, kernel: int = 0, eps: float = 0):
        """Normalizes the array and calculates sample distances.

        Args:
            method (str): One of the values from `methods`. The method used for sample distance calculation.
            kernel (int): Averaging kernel size. This kernel is applied to the fractions.
            eps (float): epsilon added to the intensities to overcome problems with zeros.


        """
        self.normalize_array_with_kernel(kernel, eps)
        self.calc_distances(method)
        self._unset_scores_and_pvalues()

    def determine_peaks(self):
        """Determines the Mean Distance, Peak Positions and shift direction.

        The Peaks are determined the following way:

        #. Calculate the mean of the :attr:`norm_array` per group (RNase & Control)
        #. Calculate the mixture distribution of the mean distributions.
        #. Take the max of the relative entropy between the mean distribution and the mixture distribution
        #. This yields a peak per group

        This so-called peak reflect the highest change in probability mass compared to the other group.

        The Mean Distance is just the Jensen-Shannon-Distance between the mean distributions

        """
        indices = self.internal_design_matrix.groupby("RNase", group_keys=True).apply(lambda x: list(x.index))
        rnase_false = self.norm_array[:, indices[False]].mean(axis=-2)
        rnase_true = self.norm_array[:, indices[True]].mean(axis=-2)
        mid = 0.5 * (rnase_true + rnase_false)
        rel1 = rel_entr(rnase_false, mid)
        r1 = np.argmax(rel1, axis=-1)
        r1 = r1 + int(np.ceil(self.state.kernel_size / 2))
        self.df["RNase False peak pos"] = r1

        rel2 = rel_entr(rnase_true, mid)
        r2 = np.argmax(rel2, axis=-1)
        jsd = jensenshannon(rnase_true, rnase_false, axis=-1, base=2)

        r2 = r2 + int(np.ceil(self.state.kernel_size / 2))
        self.df["RNase True peak pos"] = r2
        self.df["Mean Distance"] = jsd
        side = r1 - r2
        side[side < 0] = -1
        side[side > 0] = 1
        shift_strings = np.empty(side.shape, dtype='U10')
        shift_strings = np.where(side == 0, "no shift", shift_strings)
        shift_strings = np.where(side == -1, "right", shift_strings)
        shift_strings = np.where(side == 1, "left", shift_strings)
        self.df["shift direction"] = shift_strings

    def calc_cluster_features(self, kernel_range: int = 2):
        if "shift direction" not in self.df:
            raise ValueError("Peaks not determined. Determine Peaks first")
        rnase_false = self.norm_array[:, self._indices_false].mean(axis=-2)
        rnase_true = self.norm_array[:, self._indices_true].mean(axis=-2)
        mixture = 0.5 * (rnase_true + rnase_false)
        ctrl_peak = rel_entr(rnase_false, mixture)
        rnase_peak = rel_entr(rnase_true, mixture)
        ctrl_peak_pos = (self.df["RNase False peak pos"] - int(np.floor(self.state.kernel_size / 2)) - 1).to_numpy()
        rnase_peak_pos = (self.df["RNase True peak pos"] - int(np.floor(self.state.kernel_size / 2)) - 1).to_numpy()

        ctrl_peak = np.pad(ctrl_peak, ((0, 0), (kernel_range, kernel_range)), constant_values=0)
        ctrl_peak_range = np.stack((ctrl_peak_pos, ctrl_peak_pos + 2 * kernel_range + 1), axis=1)
        ctrl_peak_range = np.apply_along_axis(lambda m: np.arange(start=m[0], stop=m[1]), arr=ctrl_peak_range, axis=-1)
        v1 = np.take_along_axis(ctrl_peak, ctrl_peak_range, axis=-1)

        rnase_peak = np.pad(rnase_peak, ((0, 0), (kernel_range, kernel_range)), constant_values=0)
        rnase_peak_range = np.stack((rnase_peak_pos, rnase_peak_pos + 2 * kernel_range + 1), axis=1)
        rnase_peak_range = np.apply_along_axis(lambda m: np.arange(start=m[0], stop=m[1]), arr=rnase_peak_range,
                                               axis=-1)
        v2 = np.take_along_axis(rnase_peak, rnase_peak_range, axis=-1)
        shift = ctrl_peak_pos - rnase_peak_pos
        cluster_values = np.concatenate((shift[:, np.newaxis], v1, v2), axis=1)
        self.cluster_features = cluster_values
        self.state.cluster_kernel_distance = kernel_range

    def reduce_dim(self, data, embedding_dim: int = 2, method: str = "T-SNE"):
        data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
        if method == "T-SNE":
            reducer = TSNE(
                n_components=embedding_dim,
                perplexity=10,
                init="random",
                n_iter=250,
                random_state=0,
                method="exact" if embedding_dim >= 4 else "barnes_hut"
            )
        elif method == "UMAP":
            reducer = UMAP(n_components=embedding_dim)
        elif method == "PCA":
            reducer = PCA(n_components=embedding_dim)
        else:
            raise NotImplementedError("Method not implemented")
        embedding = np.zeros((self.array.shape[0], embedding_dim))
        mask = ~np.isnan(self.cluster_features).any(axis=1)
        embedding[mask, :] = reducer.fit_transform(data[mask])
        embedding[~mask] = np.nan
        return embedding

    def set_embedding(self, dim, method):
        self.current_embedding = self.reduce_dim(data=self.cluster_features, method=method, embedding_dim=dim)
        self.state.dimension_reduction = method

    def remove_clusters(self):
        if "Cluster" in self.df:
            self.df = self.df.drop("Cluster", axis=1)
        self.cluster_features = None

    def cluster_data(self, method: str = "HDBSCAN", **kwargs):
        if self.cluster_features is None:
            raise ValueError("Cluster Features not calculated. Calculate first")
        data = self.cluster_features
        data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
        if method == "HDBSCAN":
            clusterer = HDBSCAN(**kwargs)
        elif method == "K-Means":
            clusterer = KMeans(n_init="auto", **kwargs)
        elif method == "DBSCAN":
            clusterer = DBSCAN(**kwargs)
        else:
            raise ValueError("Unsupported Method selected")

        clusters = np.empty(self.array.shape[0])
        mask = ~np.isnan(data).any(axis=1)
        clusters[mask] = clusterer.fit(data[mask]).labels_
        clusters[~mask] = np.nan
        self.df["Cluster"] = clusters
        self.state.cluster_method = method
        self.state.cluster_args = kwargs
        return clusters

    @staticmethod
    def _jensenshannondistance(array) -> np.ndarray:
        return jensenshannon(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1), base=2, axis=-2)

    @staticmethod
    def _symmetric_kl_divergence(array):
        r1 = rel_entr(array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1)).sum(axis=-2)
        r2 = rel_entr(array[:, :, :, None].transpose(0, 3, 2, 1), array[:, :, :, None]).sum(axis=-2)
        return r1 + r2

    @staticmethod
    def _euclidean_distance(array):
        return np.linalg.norm(array[:, :, :, None] - array[:, :, :, None].transpose(0, 3, 2, 1), axis=-2)

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

    def _get_innergroup_distances(self, indices_false, indices_true):
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

    def calc_welchs_t_test(self, distance_cutoff: float = None):
        """Runs Welchs T-Test at RNase and control peak position.
        The p-Values are adjusted for multiple testing.

        .. warning::
            Since you are dealing with multivariate data, this is not the recommended way to calculate p-Values.
            Instead, use a PERMANOVA if you have a sufficient amount of replicates or consider ranking the Table using
            values calculated via the :func:`~calc_all_scores` function.

        Args:
            distance_cutoff (float): P-Values are not Calculated for proteins with a mean distance below this threshold.
                This reduces number of tests.
        """
        if "RNase True peak pos" not in self.df:
            raise ValueError("Need to compute peak positions first")
        for peak, name in (
                ("RNase True peak pos", "RNase Peak adj p-Value"), ("RNase False peak pos", "CTRL Peak adj p-Value")):
            idx = np.asarray(self.df[peak] - int(np.ceil(self.state.kernel_size / 2)))
            t = np.take_along_axis(self.norm_array, idx[:, np.newaxis, np.newaxis], axis=2).squeeze()
            t_idx = np.tile(np.asarray(self._indices_true), t.shape[0]).reshape(t.shape[0], -1)
            f_idx = np.tile(np.asarray(self._indices_false), t.shape[0]).reshape(t.shape[0], -1)
            true = np.take_along_axis(t, t_idx, axis=-1)
            false = np.take_along_axis(t, f_idx, axis=-1)
            t_test = ttest_ind(true, false, axis=1, equal_var=False)
            adj_pval = np.zeros(t_test.pvalue.shape)
            mask = np.isnan(t_test.pvalue)
            if distance_cutoff is not None:
                if "Mean Distance" not in self.df.columns:
                    raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
                mask[self.df["Mean Distance"] < distance_cutoff] = True
            adj_pval[mask] = np.nan
            _, adj_pval[~mask], _, _ = multipletests(t_test.pvalue[~mask], method="fdr_bh")
            self.df[name] = adj_pval

    def rank_table(self, values, ascending):
        """Ranks the :attr:`df`

        This can be useful if you don´t have a sufficient number of samples and thus can`t calculate a p-value.
        The ranking scheme can be set via the function parameters.

        Args:
            values (List[str]): which columns to use for ranking
            ascending (List[bool]): a boolean list indicating whether the column at the same index in values should
                be sorted ascending.

        """
        if not all([value in self.df.columns for value in values]):
            raise ValueError("Not all values that are specified in ranking scheme are already calculated")
        rdf = self.df.sort_values(values, ascending=ascending)[["RDPMSpecID"]]
        rdf["Rank"] = np.arange(1, len(rdf) + 1)
        self.df = self.df.reset_index(drop=True).merge(rdf, how="left", on="RDPMSpecID").set_index("id")
        self.df["id"] = self.df.index

    def calc_all_scores(self):
        """Calculates ANOSIM R, shift direction, peak positions and Mean Sample Distance.

        """
        self.calc_all_anosim_value()
        self.determine_peaks()

    def _calc_anosim(self, indices_false, indices_true):
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

    def _calc_permanova_f(self, indices_false, indices_true):
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
        f = (ssa) / (ssw / (bn - 2))
        return f

    def calc_all_permanova_f(self):
        """Calculates PERMANOVA F for each protein and stores it in :py:attr:`df`
        """
        f = self._calc_permanova_f(self._indices_false, self._indices_true)
        self.df["PERMANOVA F"] = f.round(decimals=DECIMALS)
        self.state.permanova_f = True


    def calc_all_anosim_value(self):
        """Calculates ANOSIM R for each protein and stores it in :py:attr:`df`"""
        r = self._calc_anosim(self._indices_false, self._indices_true)
        self.df["ANOSIM R"] = r.round(decimals=DECIMALS)
        self.state.anosim_r = True

    def _calc_global_anosim_distribution(self, nr_permutations: int, threads: int, seed: int = 0):
        np.random.seed(seed)
        _split_point = len(self._indices_false)
        indices = np.concatenate((self._indices_false, self._indices_true))
        calls = []
        for _ in range(nr_permutations):
            shuffled = np.random.permutation(indices)
            calls.append((shuffled[:_split_point], shuffled[_split_point:]))

        with multiprocessing.Pool(threads) as pool:
            result = pool.starmap(self._calc_anosim, calls)
        result = np.stack(result)
        self._anosim_distribution = result

    def _calc_global_permanova_distribution(self, nr_permutations: int, threads: int, seed: int = 0):
        np.random.seed(seed)
        _split_point = len(self._indices_false)
        indices = np.concatenate((self._indices_false, self._indices_true))
        calls = []
        for _ in range(nr_permutations):
            shuffled = np.random.permutation(indices)
            calls.append((shuffled[:_split_point], shuffled[_split_point:]))

        with multiprocessing.Pool(threads) as pool:
            result = pool.starmap(self._calc_permanova_f, calls)
        result = np.stack(result)
        self._permanova_distribution = result

    def calc_anosim_p_value(self, permutations: int, threads: int, seed: int = 0, distance_cutoff: float = None,
                            mode: str = "local"):
        """Calculates ANOSIM p-value via shuffling and stores it in :attr:`df`.
        Adjusts for multiple testing.

        Args:
            permutations (int): number of permutations used to calculate p-value
            threads (int): number of threads used for calculation
            seed (int): seed for random permutation
            distance_cutoff (float): reduces number of tests via testing only proteins with mean distance above threshold.
            mode (str): either local or global. Global uses distribution of R value of all proteins as background.
                Local uses protein specific distribution.
        """
        if "ANOSIM R" not in self.df.columns:
            self.calc_all_anosim_value()
        self._calc_global_anosim_distribution(permutations, threads, seed)
        distribution = self._anosim_distribution

        r_scores = self.df["ANOSIM R"].to_numpy()
        if mode == "global":
            distribution = distribution.flatten()
            distribution = distribution[~np.isnan(distribution)]
            p_values = np.asarray(
                [np.count_nonzero(distribution >= r_score) / distribution.shape[0] for r_score in r_scores]
            )
        elif mode == "local":
            p_values = np.count_nonzero(distribution >= r_scores, axis=0) / distribution.shape[0]
        mask = self.df["ANOSIM R"].isna()
        if distance_cutoff is not None:
            if "Mean Distance" not in self.df.columns:
                raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
            mask[self.df["Mean Distance"] < distance_cutoff] = True
        p_values[mask] = np.nan
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df[f"{mode} ANOSIM adj p-Value"] = p_values

    def calc_permanova_p_value(self, permutations: int, threads: int, seed: int = 0, distance_cutoff: float = None,
                               mode: str = "local"):
        """Calculates PERMANOVA p-value via shuffling and stores it in :attr:`df`.
        Adjusts for multiple testing.

        Args:
            permutations (int): number of permutations used to calculate p-value
            threads (int): number of threads used for calculation
            seed (int): seed for random permutation
            distance_cutoff (float): reduces number of tests via testing only proteins with mean distance above threshold.
            mode (str): either local or global. Global uses distribution of pseudo F value of all proteins as background.
                Local uses protein specific distribution.
        """
        if "PERMANOVA F" not in self.df.columns:
            self.calc_all_permanova_f()
        self._calc_global_permanova_distribution(permutations, threads, seed)
        distribution = self._permanova_distribution
        f_scores = self.df["PERMANOVA F"].to_numpy()
        if mode == "global":
            distribution = distribution.flatten()
            distribution = distribution[~np.isnan(distribution)]
            p_values = np.asarray(
                [np.count_nonzero(distribution >= f_score) / distribution.shape[0] for f_score in f_scores]
            )
        elif mode == "local":
            p_values = np.count_nonzero(distribution >= f_scores, axis=0) / distribution.shape[0]
        mask = self.df["PERMANOVA F"].isna()
        if distance_cutoff is not None:
            if "Mean Distance" not in self.df.columns:
                raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
            mask[self.df["Mean Distance"] < distance_cutoff] = True
        p_values[mask] = np.nan
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df[f"{mode} PERMANOVA adj p-Value"] = p_values
        self.state.permanova = mode
        self.state.permanova_permutations = permutations
        self.state.permanova_cutoff = distance_cutoff

    def export_csv(self, file: str, sep: str = ","):
        """Exports the :attr:`extra_df` to a file.

        Args:
            file (str): Path to file where dataframe should be exported to.
            sep (str): seperator to use.

        """
        df = self.extra_df.drop(["id"], axis=1)
        df.to_csv(file, sep=sep, index=False)

    def to_jsons(self):
        s = json.dumps(self, cls=RDPMSpecEncoder)
        return s

    def to_json(self, file: str):
        """Exports the object to JSON

         Args:
            file (str): Path to the file where the JSON encoded object should be stored.
        """
        s = self.to_jsons()
        with open(file, "w") as handle:
            handle.write(s)

    @classmethod
    def from_json(cls, json_string):
        json_obj = json.loads(json_string)
        data = cls._from_dict(json_obj)
        return data

    @classmethod
    def _from_dict(cls, dict_repr):

        for key, value in dict_repr.items():
            if key == "state":
                dict_repr[key] = RDPMState(**value)
            elif key in ("df", "design", "internal_design_matrix"):
                dict_repr[key] = pd.read_json(value).round(decimals=DECIMALS).fillna(value=np.nan)
            elif isinstance(value, list):
                dict_repr[key] = np.asarray(value)
                if isinstance(dict_repr[key], np.floating):
                    dict_repr[key] = dict_repr[key].round(decimals=DECIMALS)
            elif value == "true":
                dict_repr[key] = True
            elif value == "false":
                dict_repr[key] = False
        data = cls(dict_repr["df"], design=dict_repr["design"], logbase=dict_repr["logbase"])
        for key, value in dict_repr.items():
            setattr(data, key, value)
        return data








class RDPMSpecEncoder(JSONEncoder):
    def default(self, obj_to_encode):
        if isinstance(obj_to_encode, pd.DataFrame):
            return obj_to_encode.to_json(double_precision=15)
            # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, RDPMSpecData):
            return obj_to_encode.__dict__
        if hasattr(obj_to_encode, 'to_json'):
            return obj_to_encode.to_json()
        if isinstance(obj_to_encode, np.bool_):
            return super().encode(bool(obj_to_encode))
        return obj_to_encode.__dict__


def _analysis_executable_wrapper(args):
    rdpmspec = RDPMSpecData.from_files(args.input, args.design_matrix, sep=args.sep, logbase=args.logbase)
    kernel_size = args.kernel_size if args.kernel_size > 0 else 0
    rdpmspec.normalize_and_get_distances(args.distance_method, kernel_size, args.eps)
    rdpmspec.calc_all_scores()
    if args.distance_method is not None:
        if not args.global_permutation:
            if args.distance_method.upper() == "PERMANOVA":
                rdpmspec.calc_permanova_p_value(args.permutations, args.num_threads, mode="local")
            elif args.distance_method.upper() == "ANOSIM":
                rdpmspec.calc_anosim_p_value(args.permutations, args.num_threads, mode="local")
        else:
            if args.distance_method.upper() == "PERMANOVA":
                rdpmspec.calc_permanova_p_value(args.permutations, args.num_threads, mode="global")
            elif args.distance_method.upper() == "ANOSIM":
                rdpmspec.calc_anosim_p_value(args.permutations, args.num_threads, mode="global")
    rdpmspec.export_csv(args.output, str(args.sep))


if __name__ == '__main__':
    df = pd.read_csv("../testData/rdeep_counts_normalized.tsv", sep="\t", index_col=0)
    # sdf = df[[col for col in df.columns if "LFQ" in col]]
    sdf = df
    sdf.index = sdf.index.astype(str)
    design = pd.read_csv("../testData/rdeep_design_normalized.tsv", sep="\t")
    rdpmspec = RDPMSpecData(sdf, design)
    rdpmspec.normalize_and_get_distances("jensenshannon", 3)
    rdpmspec.calc_all_scores()
    rdpmspec.calc_cluster_features(kernel_range=3)
    clusters = rdpmspec.cluster_data()
    embedding = rdpmspec.reduce_dim()
    import plotly.graph_objs as go
    from plotly.colors import qualitative
    from RDPMSpecIdentifier.plots import plot_dimension_reduction_result

    fig = plot_dimension_reduction_result(embedding, rdpmspec, colors=qualitative.Light24 + qualitative.Dark24,
                                          clusters=clusters, name="bla")
    fig.show()
    exit()
    rdpmspec.calc_anosim_p_value(100, threads=2, mode="global")
    rdpmspec.calc_permanova_p_value(100, threads=2, mode="global")
    rdpmspec.rank_table(["ANOSIM R"], ascending=(True,))
    # rdpmspec.calc_welchs_t_test()
