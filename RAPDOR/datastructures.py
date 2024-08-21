import copy
import multiprocessing
import os
import time
import itertools as it
from functools import cached_property
import pandas as pd
from pandas.api.types import is_float_dtype
import numpy as np
from typing import Callable
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from scipy.stats import entropy
# from RAPDOR.stats import fit_ecdf, get_permanova_results
from multiprocessing import Pool
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN, DBSCAN
# from umap import UMAP
from dataclasses import dataclass
import json
from json import JSONEncoder
from pandas.testing import assert_frame_equal
from statsmodels.distributions.empirical_distribution import ECDF
import warnings
from io import StringIO





DECIMALS = 15





def check_equality(value, other_item):
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
    elif isinstance(value, list) or isinstance(value, tuple):
        if not (all([check_equality(v, other_item[idx]) for idx, v in enumerate(value)])):
            return False
    else:
        if value != other_item:
            return False
    return True


@dataclass()
class RAPDORState:
    distance_method: str = None
    kernel_size: int = None
    beta: float = None
    eps: float = None
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
        if not isinstance(other, RAPDORState):
            return False
        else:
            other_dict = other.__dict__
            for key, value in self.__dict__.items():
                if value != other_dict[key]:
                    return False
            return True


class RAPDORData:
    r""" The RAPDORData Class storing results and containing functions for analysis

     Attributes:
        df (pd.Dataframe): the dataframe that stores intensities and additional columns per protein.
        logbase (int): the logbase if intensities in :attr:`df` are log transformed. Else None.
        design (pd.Dataframe): dataframe containing information about the intensity columns in :attr:`df`
        array: (np.ndarray): The non-normalized intensities from the :attr:`df` intensity columns.
        min_replicates (int): minimum number of replicates required to calculate scores
        internal_design_matrix (pd.Dataframe): dataframe where fraction columns are stored as a list instead of
            seperate columns
        norm_array (Union[None, np.ndarray]): An array containing normalized values that add up to 1.
        distances (Union[None, np.ndarray]): An array of size `num_proteins x num_samples x num_samples` that stores the
            distance between samples. If no distance was calculated it is None.
        permutation_sufficient_samples (bool): Set to true if there are at least 5 samples per condition. Else False.
        score_columns (List[str]): list of strings that are used as column names for scores that can be calculated via
            this object.
        control (str): Name of the level of treatment that should be used as control.


     Examples:
        An instance of the RAPDORData class is obtained via the following code. Make sure your csv files
        are correctly fomatted as desribed in the :ref:`Data Prepatation<data-prep-tutorial>` Tutorial.

        >>> df = pd.read_csv("../testData/testFile.tsv", sep="\t")
        >>> design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
        >>> rapdor = RAPDORData(df, design, logbase=2)
    """
    methods = [
        "Jensen-Shannon-Distance",
        "KL-Divergence",
        "Euclidean-Distance",
    ]
    score_columns = [
        "Rank",
        "ANOSIM R",
        "global ANOSIM adj p-Value",
        "global ANOSIM raw p-Value",
        "local ANOSIM adj p-Value",
        "local ANOSIM raw p-Value",
        "PERMANOVA F",
        "global PERMANOVA adj p-Value",
        "global PERMANOVA raw p-Value",
        "local PERMANOVA adj p-Value",
        "local PERMANOVA raw p-Value",
        "Mean Distance",
        "shift direction",
        "relative fraction shift",
        "Permanova p-value",
        "Permanova adj-p-value",
        "CTRL Peak adj p-Value",
        "RNase Peak adj p-Value",
        "position strongest shift"
    ]
    replicate_info = [
        "min replicates per group",
        "contains empty replicate",
    ]

    _id_columns = ["RAPDORid", "id"]

    # prevents setting these attributes when loading from json
    _blacklisted_fields = [
        "internal_design_matrix",
        "_data_cols",
        "indices",
        "control",
        "measure",
        "measure_type"
    ]

    def __init__(
            self,
            df: pd.DataFrame,
            design: pd.DataFrame,
            logbase: int = None,
            min_replicates: int = 2,
            control: str = "Control",
            measure_type: str = "Protein",
            measure: str = "Intensities"
    ):
        self.state = RAPDORState()
        self.df = df
        self.logbase = logbase
        self.design = design
        self.min_replicates = min_replicates
        self.control = control
        self.measure_type = measure_type
        self.measure = measure
        if self.min_replicates < 2:
            raise ValueError("A minimum of two replicates is required to run statistics")
        self.array = None
        self.internal_design_matrix = None
        self.fractions = None
        self.norm_array = None
        self.kernel_array = None
        self.distances = None
        self._data_cols = None
        self._current_eps = None
        self.indices = None
        self.cluster_features = None
        self.current_embedding = None
        self.permutation_sufficient_samples = False
        self._check_design()
        self._check_dataframe()
        self._set_design_and_array()

    def __eq__(self, other):
        if not isinstance(other, RAPDORData):
            return False
        else:
            other_dict = other.__dict__
            for key, value in self.__dict__.items():
                other_item = other_dict[key]
                v = check_equality(value, other_item)
                if not v:
                    return False
            return True

    def __getitem__(self, item):
        """
        Args:
            item (List[str]): RAPDORid list

        Returns:  pd.Index
            The index of the RAPDORid list matching the ordering. Note if the RAPDORid list is not unique the length
            of the returned indices will be longer than the requested list

        """
        proteins = self.df[self.df.loc[:, "RAPDORid"].isin(item)]
        proteins["Value"] = pd.Categorical(proteins['RAPDORid'], categories=item, ordered=True)
        indices = proteins.sort_values(by='Value').index
        return indices

    def _check_dataframe(self):
        if not set(self.design["Name"]).issubset(set(self.df.columns)):
            raise ValueError("Not all Names in the designs Name column are columns in the count df")

    def _check_design(self):
        for col in ["Fraction", "Treatment", "Replicate", "Name"]:
            if not col in self.design.columns:
                raise IndexError(f"{col} must be a column in the design dataframe\n")

    def _set_design_and_array(self):
        design_matrix = self.design
        treatment_levels = sorted(design_matrix["Treatment"].unique().tolist())
        if self.control in treatment_levels:
            treatment_levels.remove(self.control)
            treatment_levels = [self.control] + treatment_levels

        design_matrix["Treatment"] = pd.Categorical(design_matrix["Treatment"], categories=treatment_levels,
                                                    ordered=True)
        self.score_columns += [f"{treatment} expected shift" for treatment in treatment_levels]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.treatment_levels = design_matrix["Treatment"].unique().sort_values().to_list()
        if len(self.treatment_levels) != 2:
            raise ValueError(f"Number of treatment levels is not equal 2:\n found levels are {self.treatment_levels}")
        design_matrix = design_matrix.sort_values(by=["Fraction", "Treatment", "Replicate"])
        tmp = design_matrix.groupby(["Treatment", "Replicate"], as_index=False, observed=False)["Name"].agg(
            list).dropna().reset_index()
        self.df.index = np.arange(self.df.shape[0])
        self.df = self.df.round(decimals=DECIMALS)
        self.fractions = design_matrix["Fraction"].unique()
        self.categorical_fraction = self.fractions.dtype == np.dtype('O')
        self.permutation_sufficient_samples = bool(
            np.all(tmp.groupby("Treatment", observed=False)["Replicate"].count() >= 5))
        l = []
        rnames = []
        for idx, row in tmp.iterrows():
            rnames += row["Name"]
        self._data_cols = np.asarray(rnames)
        for col in self._data_cols:
            self.df.loc[:,col] = pd.to_numeric(self.df.loc[:, col], errors="coerce")

        for idx, row in tmp.iterrows():
            sub_df = self.df[row["Name"]].to_numpy(dtype=float)
            l.append(sub_df)
        self.df["RAPDORid"] = self.df.iloc[:, 0]
        self.df["id"] = self.df.index
        array = np.stack(l, axis=1)
        if self.logbase is not None:
            array = np.power(self.logbase, array)
        mask = np.isnan(array)
        array[mask] = 0
        self.array = array
        self.internal_design_matrix = tmp
        indices = self.internal_design_matrix.groupby("Treatment", group_keys=True, observed=False).apply(
            lambda x: list(x.index), include_groups=False)
        self.indices = tuple(np.asarray(index) for index in indices)

        p = ~np.any(self.array, axis=-1)
        pf = p[:, self.indices[0]]
        pf = pf.shape[-1] - np.count_nonzero(pf, axis=-1)

        pt = p[:, self.indices[1]]
        pt = pt.shape[-1] - np.count_nonzero(pt, axis=-1)

        tmp = np.any(p, axis=-1)
        self.df["contains empty replicate"] = tmp
        self.df["min replicates per group"] = np.min(np.stack((pt, pf), axis=-1), axis=-1)

    @cached_property
    def raw_lfc(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false = np.nansum(np.nanmean(self.array[:, self.indices[0]], axis=-2),axis=-1)
            rnase_true = np.nansum(np.nanmean(self.array[:, self.indices[1]], axis=-2), axis=-1)
        ret = np.log2(rnase_true/rnase_false)
        return ret


    @classmethod
    def from_files(cls, intensities: str, design: str, logbase: int = None, sep: str = ","):
        """Constructor to generate instance from files instead of pandas dataframes.

        Args:
            intensities (str): Path to the intensities File
            design (str): Path to the design file
            logbase (Union[None, int]): Logbase if intensities in the intensity file are log transformed
            sep (str): seperator used in the intensities and design files. Must be the same for both.

        Returns: RAPDORData

        """
        design = pd.read_csv(design, sep=sep)
        df = pd.read_csv(intensities, sep=sep, index_col=0)
        df.index = df.index.astype(str)
        rapdor = RAPDORData(df, design, logbase)
        return rapdor

    @property
    def extra_df(self):
        """ Return a Dataframe Slice all columns from self.df that are not part of the intensity columns

        Returns: pd.Dataframe

        """
        if self._data_cols is None:
            return None
        return self.df.iloc[:, ~np.isin(self.df.columns, self._data_cols)]

    @staticmethod
    def _normalize_rows(array, eps: float = 0):
        if eps:
            array += eps
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            array = array / np.sum(array, axis=-1, keepdims=True)
        array = array.round(DECIMALS)
        return array

    @cached_property
    def _treatment_means(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false = np.nanmean(self.norm_array[:, self.indices[0]], axis=-2)
            rnase_true = np.nanmean(self.norm_array[:, self.indices[1]], axis=-2)
        ret = np.stack((rnase_false, rnase_true))
        return ret

    def _del_treatment_means(self):
        if "_treatment_means" in self.__dict__:
            del self.__dict__["_treatment_means"]

    def normalize_array_with_kernel(self, kernel_size: int = 0, eps: float = 0):
        """Normalizes the array and sets `norm_array` attribute.

        Args:
            kernel_size (int): Averaging kernel size. This kernel is applied to the fractions.
            eps (float): epsilon added to the intensities to overcome problems with zeros.

        """
        array = self.array
        self._del_treatment_means()

        if kernel_size:
            if not kernel_size % 2:
                raise ValueError(f"Kernel size must be odd")
            kernel = np.ones(kernel_size) / kernel_size
            array = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=-1, arr=array)
        self.kernel_array = array
        self.norm_array = self._normalize_rows(array, eps=eps)
        self.state.kernel_size = kernel_size
        self.state.eps = eps


    def _calc_distance_via(self, method, array1, array2, axis: int = -1):
        if method == "Jensen-Shannon-Distance":
            distances = self._jensenshannondistance(array1, array2, axis=axis)

        elif method == "KL-Divergence":
            if self.state.eps is None or self.state.eps <= 0:
                raise ValueError(
                    "Cannot calculate KL-Divergence for Counts with 0 entries. "
                    "Need to set epsilon which is added to the raw Protein intensities"
                )
            distances = self._symmetric_kl_divergence(array1, array2, axis=axis)
        elif method == "Euclidean-Distance":
            distances = self._euclidean_distance(array1, array2, axis=axis)
        else:
            raise ValueError(f"methhod: {method} is not supported")
        return distances

    def calc_distances(self, method: str):
        """Calculates between sample distances.
                
        Args:
            method (str): One of the values from `methods`. The method used for sample distance calculation.

        Raises:
            ValueError: If the method string is not supported or symmetric-kl-divergence is used without adding an
                epsilon to the protein intensities

        """
        array1, array2 = self.norm_array[:, :, :, None], self.norm_array[:, :, :, None].transpose(0, 3, 2, 1)
        self.distances = self._calc_distance_via(method, array1=array1, array2=array2, axis=-2)
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

    def determine_strongest_shift(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false, rnase_true = self._treatment_means
            mid = 0.5 * (rnase_true + rnase_false)
            if self.state.distance_method in ["Jensen-Shannon-Distance", "KL-Divergence"]:
                rel1 = rel_entr(rnase_false, mid)
                rel2 = rel_entr(rnase_true, mid)
            elif self.state.distance_method == "Euclidean-Distance":
                rel1 = rnase_false - mid
                rel2 = rnase_true - mid
            else:
                raise ValueError(f"Peak determination failed due to bug in source code")
            test = rel2
            if self.categorical_fraction:
                positions = np.argmax(test, axis=-1)
                positions += self.state.kernel_size // 2
                positions = np.asarray(self.fractions)[positions]
            else:

                i = self.state.kernel_size // 2
                t = ((test == np.max(test, axis=-1, keepdims=True)) * self.fractions[i:-i])
                positions = t.sum(axis=-1) / np.count_nonzero(t, axis=-1)
                positions = np.floor(positions).astype(int)
        # Get the middle occurrence index
        self.df["position strongest shift"] = positions
        self.df.loc[self.df["Mean Distance"].isna(), "position strongest shift"] = pd.NA

    def calc_mean_distance(self):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false, rnase_true = self._treatment_means
        jsd = self._calc_distance_via(self.state.distance_method, rnase_true, rnase_false, axis=-1)
        self.df["Mean Distance"] = jsd

    def determine_peaks(self, beta: float = 1000):
        """Determines the Mean Distance, Peak Positions and shift direction.

        The Peaks are determined the following way:

        #. Calculate the mean of the :attr:`norm_array` per group (RNase & Control)
        #. Calculate the mixture distribution of the mean distributions.
        #. Calculate $D$ which is either:
            * Relative position-wise entropy of both groups to the mixture distribution if distance method is KL-Divergence or Jensen-Shannon
            * position-wise euclidean distance of both groups to the mixture distribution if distance method is Eucledian-Distance
        #. Apply a soft-argmax to this using beta hyperparameter to find the relative position shift

        """
        self.state.beta = beta
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false, rnase_true = self._treatment_means
        mid = 0.5 * (rnase_true + rnase_false)
        s = int(np.ceil(self.state.kernel_size / 2))
        range = np.arange(s, s + mid.shape[-1])
        if self.state.distance_method in ["Jensen-Shannon-Distance", "KL-Divergence"]:
            rel1 = rel_entr(rnase_false, mid)
            rel2 = rel_entr(rnase_true, mid)
        elif self.state.distance_method == "Euclidean-Distance":
            rel1 = rnase_false - mid
            rel2 = rnase_true - mid
        else:
            raise ValueError(f"Peak determination failed due to bug in source code")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            # rel1[(rel1 <= 0)] = np.nan
            softmax1 = ((np.exp(beta * rel1)) / np.nansum(np.exp(beta * rel1), axis=-1, keepdims=True))
            r1 = np.nansum(softmax1 * range, axis=-1)
            self.df[f"{self.treatment_levels[0]} expected shift"] = r1.round(decimals=DECIMALS)

            # rel2[(rel2 <= 0)] = np.nan
            softmax2 = ((np.exp(beta * rel2)) / np.nansum(np.exp(beta * rel2), axis=-1, keepdims=True))
            r2 = np.nansum(softmax2 * range, axis=-1)
            # r2 = np.nanargmax(rel2, axis=-1)

            self.df[f"{self.treatment_levels[1]} expected shift"] = r2.round(decimals=DECIMALS)
        side = r2 - r1
        self.df["relative fraction shift"] = side
        side[side < 0] = -1
        side[side > 0] = 1
        shift_strings = np.empty(side.shape, dtype='U10')
        shift_strings = np.where(side == 0, "no direction", shift_strings)
        shift_strings = np.where(side == -1, "left", shift_strings)
        shift_strings = np.where(side == 1, "right", shift_strings)
        self.df["shift direction"] = shift_strings

    # def calc_cluster_features(self, kernel_range: int = 2):
    #     if "shift direction" not in self.df:
    #         raise ValueError("Peaks not determined. Determine Peaks first")
    #     rnase_false = self.norm_array[:, self._indices_false].mean(axis=-2)
    #     rnase_true = self.norm_array[:, self._indices_true].mean(axis=-2)
    #     mixture = 0.5 * (rnase_true + rnase_false)
    #     ctrl_peak = rel_entr(rnase_false, mixture)
    #     rnase_peak = rel_entr(rnase_true, mixture)
    #     ctrl_peak_pos = (self.df["RNase False peak pos"] - int(np.floor(self.state.kernel_size / 2)) - 1).to_numpy()
    #     rnase_peak_pos = (self.df["RNase True peak pos"] - int(np.floor(self.state.kernel_size / 2)) - 1).to_numpy()
    #
    #     ctrl_peak = np.pad(ctrl_peak, ((0, 0), (kernel_range, kernel_range)), constant_values=0)
    #     ctrl_peak_range = np.stack((ctrl_peak_pos, ctrl_peak_pos + 2 * kernel_range + 1), axis=1)
    #     ctrl_peak_range = np.apply_along_axis(lambda m: np.arange(start=m[0], stop=m[1]), arr=ctrl_peak_range, axis=-1)
    #     v1 = np.take_along_axis(ctrl_peak, ctrl_peak_range, axis=-1)
    #
    #     rnase_peak = np.pad(rnase_peak, ((0, 0), (kernel_range, kernel_range)), constant_values=0)
    #     rnase_peak_range = np.stack((rnase_peak_pos, rnase_peak_pos + 2 * kernel_range + 1), axis=1)
    #     rnase_peak_range = np.apply_along_axis(lambda m: np.arange(start=m[0], stop=m[1]), arr=rnase_peak_range,
    #                                            axis=-1)
    #     v2 = np.take_along_axis(rnase_peak, rnase_peak_range, axis=-1)
    #     shift = ctrl_peak_pos - rnase_peak_pos
    #     cluster_values = np.concatenate((shift[:, np.newaxis], v1, v2), axis=1)
    #     self.cluster_features = cluster_values
    #     self.state.cluster_kernel_distance = kernel_range

    def calc_distribution_features(self):
        if "position strongest shift" not in self.df:
            raise ValueError("Peaks not determined. Determine Peaks first")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            rnase_false, rnase_true = self._treatment_means


            if self.state.distance_method in ["Jensen-Shannon-Distance", "KL-Divergence"]:
                false_uni_distance = entropy(rnase_false, axis=-1)
                true_uni_distance = entropy(rnase_true, axis=-1)
            else:
                mixture = rnase_true + rnase_false
                uni_nonzero = mixture > 0
                uniform = (np.ones((mixture.shape[0], mixture.shape[1])) / np.count_nonzero(uni_nonzero, axis=-1,
                                                                                            keepdims=True)) * uni_nonzero
                false_uni_distance = self._calc_distance_via(self.state.distance_method, rnase_false, uniform, axis=-1)
                true_uni_distance = self._calc_distance_via(self.state.distance_method, rnase_true, uniform, axis=-1)

        diff = false_uni_distance - true_uni_distance
        if self.categorical_fraction:
            shift = self.df["position strongest shift"].to_numpy()
        else:
            shift = self.df["relative fraction shift"].to_numpy()
        self.cluster_features = np.concatenate((shift[:, np.newaxis], diff[:, np.newaxis]), axis=1)
        self.current_embedding = self.cluster_features

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

    def remove_clusters(self):
        if "Cluster" in self.df:
            self.df = self.df.drop("Cluster", axis=1)
        self.cluster_features = None
        self.state.cluster_method = None
        self.state.cluster_args = None

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
    def _jensenshannondistance(array1, array2, axis: int = -2) -> np.ndarray:
        return jensenshannon(array1, array2, base=2, axis=axis)

    @staticmethod
    def _symmetric_kl_divergence(array1, array2, axis: int = -2):
        r1 = rel_entr(array1, array2).sum(axis=axis)
        r2 = rel_entr(array2, array1).sum(axis=axis)
        return r1 + r2

    @staticmethod
    def _euclidean_distance(array1, array2, axis: int = -2):
        return np.linalg.norm(array1 - array2, axis=axis)

    def _get_outer_group_distances(self, indices_false, indices_true):
        n_genes = self.distances.shape[0]
        mg1, mg2 = np.meshgrid(indices_true, indices_false)
        e = np.ones((n_genes, len(indices_false), len(indices_true)))
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
            e = np.ones((n_genes, len(idx), len(idx)))
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

    # def calc_welchs_t_test(self, distance_cutoff: float = None):
    #     """Runs Welchs T-Test at RNase and control peak position.
    #     The p-Values are adjusted for multiple testing.
    #
    #     .. warning::
    #         Since you are dealing with multivariate data, this is not the recommended way to calculate p-Values.
    #         Instead, use a PERMANOVA if you have a sufficient amount of replicates or consider ranking the Table using
    #         values calculated via the :func:`~calc_all_scores` function.
    #
    #     Args:
    #         distance_cutoff (float): P-Values are not Calculated for proteins with a mean distance below this threshold.
    #             This reduces number of tests.
    #     """
    #     if "RNase True peak pos" not in self.df:
    #         raise ValueError("Need to compute peak positions first")
    #     for peak, name in (
    #             ("RNase True peak pos", "RNase Peak adj p-Value"), ("RNase False peak pos", "CTRL Peak adj p-Value")):
    #         idx = np.asarray(self.df[peak] - int(np.ceil(self.state.kernel_size / 2)))
    #         t = np.take_along_axis(self.norm_array, idx[:, np.newaxis, np.newaxis], axis=2).squeeze()
    #         t_idx = np.tile(np.asarray(self._indices_true), t.shape[0]).reshape(t.shape[0], -1)
    #         f_idx = np.tile(np.asarray(self._indices_false), t.shape[0]).reshape(t.shape[0], -1)
    #         true = np.take_along_axis(t, t_idx, axis=-1)
    #         false = np.take_along_axis(t, f_idx, axis=-1)
    #         t_test = ttest_ind(true, false, axis=1, equal_var=False)
    #         adj_pval = np.zeros(t_test.pvalue.shape)
    #         mask = np.isnan(t_test.pvalue)
    #         if distance_cutoff is not None:
    #             if "Mean Distance" not in self.df.columns:
    #                 raise ValueError("Need to run peak position estimation before please call self.determine_peaks()")
    #             mask[self.df["Mean Distance"] < distance_cutoff] = True
    #         adj_pval[mask] = np.nan
    #         _, adj_pval[~mask], _, _ = multipletests(t_test.pvalue[~mask], method="fdr_bh")
    #         self.df[name] = adj_pval

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
        if "Rank" in self.df:
            self.df = self.df.drop('Rank', axis=1)
        rdf = self.df.sort_values(values, ascending=ascending)
        rdf["Rank"] = np.arange(1, len(rdf) + 1)
        rdf = rdf[["Rank"]]
        self.df = self.df.join(rdf)

    def calc_all_scores(self):
        """Calculates ANOSIM R, shift direction, peak positions and Mean Sample Distance.

        """
        self.calc_mean_distance()
        self.calc_all_anosim_value()
        if not self.categorical_fraction:
            self.determine_peaks()
        self.determine_strongest_shift()

    def _calc_anosim(self, indices_false, indices_true, ignore_nan: bool = True, ignore_zero_distances: bool = True):
        outer_group_distances = self._get_outer_group_distances(indices_false, indices_true)
        inner_group_distances = self._get_innergroup_distances(indices_false, indices_true)
        stat_distances = np.concatenate((outer_group_distances, inner_group_distances), axis=-1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            if ignore_nan:
                mask = np.isnan(stat_distances)
                ranks = stat_distances.argsort(axis=-1).argsort(axis=-1).astype(float)
                ranks[mask] = np.nan
                rb = np.nanmean(ranks[:, 0:outer_group_distances.shape[-1]], axis=-1)
                rw = np.nanmean(ranks[:, outer_group_distances.shape[-1]:], axis=-1)
                nonnan = np.count_nonzero(~mask, axis=-1)
                r = (rb - rw) / (nonnan / 2)
            else:
                mask = np.isnan(stat_distances).any(axis=-1)
                ranks = stat_distances.argsort(axis=-1).argsort(axis=-1)
                rb = np.mean(ranks[:, 0:outer_group_distances.shape[-1]], axis=-1)
                rw = np.mean(ranks[:, outer_group_distances.shape[-1]:], axis=-1)
                r = (rb - rw) / (ranks.shape[-1] / 2)
                r[mask] = np.nan
        r[self.df["min replicates per group"] < self.min_replicates] = np.nan
        if ignore_zero_distances:
            r[np.all(self.distances == 0, axis=(1, 2))] = np.nan
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
        f = self._calc_permanova_f(self.indices[0], self.indices[1])
        f[self.df["Mean Distance"] == 0] = np.nan

        self.df["PERMANOVA F"] = f.round(decimals=DECIMALS)
        self.state.permanova_f = True

    def calc_all_anosim_value(self):
        """Calculates ANOSIM R for each protein and stores it in :py:attr:`df`"""
        r = self._calc_anosim(self.indices[0], self.indices[1])
        r[self.df["Mean Distance"] == 0] = np.nan

        self.df["ANOSIM R"] = r.round(decimals=DECIMALS)
        self.state.anosim_r = True

    def _calc_global_anosim_distribution(self, nr_permutations: int, threads: int, seed: int = 0, callback=None):
        np.random.seed(seed)
        _split_point = len(self.indices[0])
        indices = np.concatenate((self.indices[0], self.indices[1]))
        calls = []
        if nr_permutations == -1:
            perms = it.permutations(indices)
            for shuffled in perms:
                calls.append((shuffled[:_split_point], shuffled[_split_point:]))
        else:
            for _ in range(nr_permutations):
                shuffled = np.random.permutation(indices)
                calls.append((shuffled[:_split_point], shuffled[_split_point:]))
        if threads > 1:
            with multiprocessing.Pool(threads) as pool:
                result = pool.starmap(self._calc_anosim, calls)
        else:
            if callback:
                result = []
                m_len = len(calls)
                for idx, call in enumerate(calls):
                    perc = str(int((idx * 97) / m_len))
                    callback(perc)
                    result.append(self._calc_anosim(*call))
            else:
                result = [self._calc_anosim(*call) for call in calls]
        result = np.stack(result)
        return result

    def _calc_global_permanova_distribution(self, nr_permutations: int, threads: int, seed: int = 0):
        np.random.seed(seed)
        _split_point = len(self.indices[0])
        indices = np.concatenate((self.indices[0], self.indices[1]))
        calls = []
        if nr_permutations == -1:
            perms = it.permutations(indices)
            for shuffled in perms:
                calls.append((shuffled[:_split_point], shuffled[_split_point:]))
        else:
            for _ in range(nr_permutations):
                shuffled = np.random.permutation(indices)
                calls.append((shuffled[:_split_point], shuffled[_split_point:]))
        if threads > 1:
            with multiprocessing.Pool(threads) as pool:
                result = pool.starmap(self._calc_permanova_f, calls)
        else:
            result = [self._calc_permanova_f(*call) for call in calls]
        result = np.stack(result)
        return result

    def calc_anosim_p_value(self, permutations: int, threads: int, seed: int = 0,
                            mode: str = "local", callback=None):
        """Calculates ANOSIM p-value via shuffling and stores it in :attr:`df`.
        Adjusts for multiple testing.

        Args:
            permutations (int): number of permutations used to calculate p-value. Set to -1 to use all possible distinct
                permutations
            threads (int): number of threads used for calculation
            seed (int): seed for random permutation
            mode (str): either local or global. Global uses distribution of R value of all proteins as background.
                Local uses protein specific distribution.
            callback(Callable): A callback function that receives the progress in the form of a percent string e.g. "50".
                This can be used in combination with a progress bar.
        Returns:
            p-values (np.ndarray): fdr corrected p-values for each protein
            distribution (np.ndarray): distribution of R values used to calculate p-values
        """
        if "ANOSIM R" not in self.df.columns:
            self.calc_all_anosim_value()
        o_distribution = self._calc_global_anosim_distribution(permutations, threads, seed, callback)
        r_scores = self.df["ANOSIM R"].to_numpy()
        if mode == "global":
            distribution = o_distribution.flatten()
            distribution = distribution[~np.isnan(distribution)]
            # Sort the distribution array
            distribution = np.sort(distribution)
            # Use searchsorted to find the insertion points for r_scores
            indices = np.searchsorted(distribution, r_scores, side='left')

            p_values = (len(distribution) - indices) / len(distribution)

            mask = self.df["contains empty replicate"].to_numpy()
            p_values[mask] = np.nan
        elif mode == "local":
            p_values = np.count_nonzero(o_distribution >= r_scores, axis=0) / o_distribution.shape[0]
            mask = self.df["ANOSIM R"].isna()
        else:
            raise ValueError("mode not supported")
        if callback:
            callback("100")
        mask[np.isnan(r_scores)] = True
        p_values[mask] = np.nan
        self.df[f"{mode} ANOSIM raw p-Value"] = p_values
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df[f"{mode} ANOSIM adj p-Value"] = p_values
        return p_values, o_distribution

    def calc_permanova_p_value(self, permutations: int, threads: int, seed: int = 0,
                               mode: str = "local"):
        """Calculates PERMANOVA p-value via shuffling and stores it in :attr:`df`.
        Adjusts for multiple testing.

        Args:
            permutations (int): number of permutations used to calculate p-value
            threads (int): number of threads used for calculation
            seed (int): seed for random permutation
            mode (str): either local or global. Global uses distribution of pseudo F value of all proteins as background.
                Local uses protein specific distribution.
        Returns:
            p-values (np.ndarray): fdr corrected p-values for each protein
            distribution (np.ndarray): distribution of R values used to calculate p-values
        """
        if "PERMANOVA F" not in self.df.columns:
            self.calc_all_permanova_f()
        o_distribution = self._calc_global_permanova_distribution(permutations, threads, seed)
        f_scores = self.df["PERMANOVA F"].to_numpy()
        if mode == "global":
            distribution = o_distribution.flatten()
            distribution = distribution[~np.isnan(distribution)]
            p_values = np.asarray(
                [np.count_nonzero(distribution >= f_score) / distribution.shape[0] for f_score in f_scores]
            )
            mask = self.df["contains empty replicate"].to_numpy()
            p_values[mask] = np.nan
        elif mode == "local":
            p_values = np.count_nonzero(o_distribution >= f_scores, axis=0) / o_distribution.shape[0]
            mask = self.df["PERMANOVA F"].isna()
        else:
            raise ValueError("mode not supported")
        mask[np.isnan(f_scores)] = True
        p_values[mask] = np.nan
        self.df[f"{mode} ANOSIM raw p-Value"] = p_values
        _, p_values[~mask], _, _ = multipletests(p_values[~mask], method="fdr_bh")
        self.df[f"{mode} PERMANOVA adj p-Value"] = p_values
        self.state.permanova = mode
        self.state.permanova_permutations = permutations
        return p_values, o_distribution

    def export_csv(self, file: str, sep: str = ","):
        """Exports the :attr:`extra_df` to a file.

        Args:
            file (str): Path to file where dataframe should be exported to.
            sep (str): seperator to use.

        """
        df = self.extra_df.drop(["id"], axis=1)
        df.to_csv(file, sep=sep, index=False)

    def to_jsons(self):
        s = json.dumps(self, cls=RAPDOREncoder)
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
    def from_file(cls, json_file):
        with open(json_file) as handle:
            json_string = handle.read()
        return cls.from_json(json_string)

    @classmethod
    def _from_dict(cls, dict_repr):

        for key, value in dict_repr.items():
            if key == "state":
                dict_repr[key] = RAPDORState(**value)
            elif key in ("df", "design", "internal_design_matrix"):
                value = StringIO(value)
                dict_repr[key] = pd.read_json(value).round(decimals=DECIMALS).fillna(value=np.nan)
            elif isinstance(value, list):
                if not isinstance(value[0], str) and key != "indices":
                    dict_repr[key] = np.asarray(value)
                    if isinstance(dict_repr[key], np.floating):
                        dict_repr[key] = dict_repr[key].round(decimals=DECIMALS)
                else:
                    dict_repr[key] = value
            elif value == "true":
                dict_repr[key] = True
            elif value == "false":
                dict_repr[key] = False
        data = cls(
            dict_repr["df"],
            design=dict_repr["design"],
            logbase=dict_repr["logbase"],
            control=dict_repr["control"],
            measure=dict_repr["measure"],
            measure_type=dict_repr["measure_type"]
        )
        for key, value in dict_repr.items():
            if key not in cls._blacklisted_fields:
                setattr(data, key, value)
        return data


class RAPDOREncoder(JSONEncoder):
    def default(self, obj_to_encode):
        if isinstance(obj_to_encode, pd.DataFrame):
            return obj_to_encode.to_json(double_precision=15)
            # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        if isinstance(obj_to_encode, RAPDORData):
            return obj_to_encode.__dict__
        if hasattr(obj_to_encode, 'to_json'):
            return obj_to_encode.to_json()
        if isinstance(obj_to_encode, np.bool_):
            return super().encode(bool(obj_to_encode))
        return obj_to_encode.__dict__


def _analysis_executable_wrapper(args):
    rapdor = RAPDORData.from_files(args.input, args.design_matrix, sep=args.sep, logbase=args.logbase)
    kernel_size = args.kernel_size if args.kernel_size > 0 else 0
    rapdor.normalize_and_get_distances(args.distance_method, kernel_size, args.eps)
    rapdor.calc_all_scores()
    if args.distance_method is not None:
        if not args.global_permutation:
            if args.distance_method.upper() == "PERMANOVA":
                rapdor.calc_permanova_p_value(args.permutations, args.num_threads, mode="local")
            elif args.distance_method.upper() == "ANOSIM":
                rapdor.calc_anosim_p_value(args.permutations, args.num_threads, mode="local")
        else:
            if args.distance_method.upper() == "PERMANOVA":
                rapdor.calc_permanova_p_value(args.permutations, args.num_threads, mode="global")
            elif args.distance_method.upper() == "ANOSIM":
                rapdor.calc_anosim_p_value(args.permutations, args.num_threads, mode="global")
    rapdor.export_csv(args.output, str(args.sep))
    if args.json is not None:
        rapdor.to_json(args.json)


if __name__ == '__main__':
    f = "/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/NatureSpatial/RawData/egf_2min_raw_intensities.tsv"
    d = "/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/NatureSpatial/RawData/egf_2min_design.tsv"
    #f  = "../testData/testFile.tsv"
    #d  = "../testData/testDesign.tsv"
    df = pd.read_csv(f, sep="\t", index_col=0)

    design = pd.read_csv(d, sep="\t")
    rapdor = RAPDORData(df, design, logbase=None)
    rapdor.normalize_and_get_distances("Jensen-Shannon-Distance", 3)
    rapdor.calc_all_scores()
    s = time.time()
    rapdor.calc_anosim_p_value(999, threads=1, mode="global")
    e = time.time()
    print(e-s)
    exit()
    clusters = rapdor.cluster_data()
    embedding = rapdor.reduce_dim()
    import plotly.graph_objs as go
    from plotly.colors import qualitative
    from RAPDOR.plots import plot_dimension_reduction_result

    fig = plot_dimension_reduction_result(embedding, rapdor, colors=qualitative.Light24 + qualitative.Dark24,
                                          clusters=clusters, name="bla")
    fig.show()
    exit()
    rapdor.rank_table(["ANOSIM R"], ascending=(True,))
    # rapdor.calc_welchs_t_test()
