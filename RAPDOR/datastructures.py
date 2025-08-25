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
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors
import threading

DECIMALS = 15


IMPUTE_R_CODE = """
msnset_impute <- function(sub_df, pdata, min_rep, n_neighbors, indExpData, colnameForID = "Protein", 
                          logData = FALSE, pep_prot_data = "protein", software = "maxquant") {
    library(MSnbase)
    library(DAPAR)

    msnset <- createMSnset(
        sub_df,
        pdata,
        indExpData = indExpData,
        colnameForID = colnameForID,
        logData = logData,
        pep_prot_data = pep_prot_data,
        software = software
    )
    level <- 'peptide'
    pattern <- "Quantified"
    type <- "AtLeastOneCond"
    percent <- FALSE
    op <- ">="
    th <- min_rep
    indices <- GetIndices_MetacellFiltering(msnset, level, pattern, type, percent, op, th)
    msnset <- MetaCellFiltering(msnset, indices, "keep")$new
    imputed_data <- wrapper.impute.KNN(msnset, n_neighbors)
    imputed_counts <- exprs(imputed_data)
    imputed_sub_df <- as.data.frame(imputed_counts)
    return(imputed_sub_df)
}
"""



def _check_equality(value, other_item):
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
        if not (all([_check_equality(v, other_item[idx]) for idx, v in enumerate(value)])):
            return False
    else:
        if value != other_item:
            return False
    return True


@dataclass()
class RAPDORState:
    """ An internal class storing the current status of your analysis.

    This is mainly used for the Dash tool and not important for regular users.

    """
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
        methods (List[str]): List of supported distance functions


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
    _replicate_info = [
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
                v = _check_equality(value, other_item)
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
        self.design = design_matrix
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
        indices = self.internal_design_matrix.groupby("Treatment", group_keys=True, observed=False)["index"].apply(
            lambda x: list(x.index))
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
        """ Calculates the log2 fold change of the raw intensity means.

        Returns:
            np.ndarray: array containing the log2 fold change of the raw intensities

        """
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

        Returns:
            pd.DataFrame: slice of :attr:`df`

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

    def _impute_using_all_fractions(self, n_perc, n_neighbors: int = 10, impute_quantile: float = 0.95):
        self.calc_distance_stats()
        for treatment, idx in zip(self.treatment_levels, self.indices):
            min_rep = np.ceil(n_perc * len(idx))
            subarray = self.array[:, idx, :]
            x = self.df[f"{treatment} distance mean"]
            perc = np.nanquantile(x, impute_quantile)
            noise_mask = x > perc

            replace_array = np.full(subarray.shape, np.nan)
            for i in range(subarray.shape[-1]):
                m = (subarray[:, :, i] > 0)
                m1 = m.sum(axis=-1)
                impute = (m1 >= min_rep) & (m1 < subarray.shape[1])
                missing_all = np.any((subarray.sum(axis=-1) == 0), axis=1)
                missing_all = missing_all | noise_mask

                for j in range(subarray.shape[-2]):
                    to_impute = (impute & (subarray[:, j, i] == 0)).reshape(-1)
                    knn = NearestNeighbors(n_neighbors=n_neighbors)
                    nn_array = np.log2(subarray.reshape(-1, subarray.shape[-1] * subarray.shape[-2]) + 1)
                    mask = np.ones(subarray.shape[1:], dtype=bool)
                    mask[j, i] = 0
                    mask = mask.reshape(subarray.shape[-1] * subarray.shape[-2])
                    if ~np.any(to_impute & ~missing_all):
                        continue
                    knn.fit(nn_array[~to_impute & ~missing_all][:, mask])
                    id = self.df[self.df["old_locus_tag"] == "ssr3189"]["id"].iloc[0]
                    neighbors = knn.kneighbors(nn_array[to_impute & ~missing_all][:, mask], return_distance=False)
                    ns = nn_array[~to_impute & ~missing_all][neighbors].reshape(neighbors.shape[0], n_neighbors,
                                                                                subarray.shape[1], -1)
                    median = np.median(ns[:, :, j, i], axis=-1)

                    replace_array[(to_impute & ~missing_all), j, i] = median

            replace_array = (2 ** replace_array) - 1
            replace_array[np.isnan(replace_array)] = 0
            t = np.argwhere(~(replace_array == 0))
            tmp = pd.DataFrame({"id": t[:, 0], f"{treatment} imputed": t[:, 1:].tolist()})
            agg = tmp.groupby('id', as_index=False).agg({f"{treatment} imputed": list})
            agg[f"{treatment} imputed"] = agg[f"{treatment} imputed"].astype(str)
            self.df = self.df.merge(agg, on="id", how="left")
            self.df[f"{treatment} imputed"].loc[noise_mask] = "too noisy"
            self.array[:, idx] += replace_array
        self._impute_fix_df()


    def _impute_fix_df(self):
        for _, row in self.internal_design_matrix.iterrows():
            array = self.array[:, row["index"], :]
            if self.logbase:
                array = np.emath.logn(self.logbase, array)
            array[np.isinf(array)] = np.nan
            self.df[row["Name"]] = array


    def _impute_via_rpy(self, n_perc=0.5, n_neighbors: int = 10, impute_quantile: float=0.95):
        try:
            from rpy2.robjects import r, pandas2ri, default_converter, numpy2ri
            from rpy2.robjects.conversion import localconverter
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError("Missing optional dependency rpy2. Please install for dataimputation")
        for package in ("DAPAR", "MSnbase"):
            t = r.require(package)[0]

            if not t:
                raise ImportError(f"Unable to import optional dependency: R package {package}. "
                                  f"Please install the corresponding R package in your environment")
        r(IMPUTE_R_CODE)
        msnset_impute = r["msnset_impute"]

        self.calc_distance_stats()
        all_missing = self.array.sum(axis=-1) == 0
        tmp_internal_design = self.internal_design_matrix.explode("Name").set_index("Name")
        for treatment, idx in zip(self.treatment_levels, self.indices):
            x = self.df[f"{treatment} distance mean"]
            perc = np.nanquantile(x, impute_quantile)
            noise_mask = x > perc
            for fidx, fraction in enumerate(self.fractions):
                min_rep = float(np.ceil(n_perc * len(idx)))

                sub_array = np.log2(self.array[:, :, fidx])
                sub_array[np.isinf(sub_array)] = np.nan
                sub_df = pd.DataFrame({
                    "Protein": self.df["id"],
                })
                design = self.design[(self.design["Treatment"] == treatment) & (self.design["Fraction"] == fraction)]
                rname = {"Treatment": "Condition","Replicate": "Bio.rep"}
                design = design.rename(rname, axis=1)
                design = design[list(rname.values()) + ["Name"]]
                design["Sample.name"] = np.arange(design.shape[0])
                design["Sample.name"] = "causeRcanthandledataframes" + design["Sample.name"].astype(str)
                for hash, sname in design[["Sample.name", "Name"]].iterrows():
                    iidx = tmp_internal_design.loc[sname["Name"]]["index"]
                    sub_df[str(sname["Sample.name"])] = sub_array[:, iidx]
                design = design.drop("Name", axis=1)
                design.index = design["Sample.name"].tolist()

                snames = [col for col in sub_df.columns[1:]]
                sub_df = sub_df[ snames + ["Protein"]]
                mask = np.all(np.isnan(sub_df[snames]), axis=1) | noise_mask
                sub_df = sub_df[~mask]
                if len(sub_df) == 0:
                    continue
                with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                    sub_df_r = pandas2ri.py2rpy(sub_df)
                    pdata_r = pandas2ri.py2rpy(design)

                    result = msnset_impute(sub_df_r, pdata_r, logData=False, min_rep=float(min_rep), n_neighbors=int(n_neighbors), indExpData=r.seq(1,sub_df.shape[1]-1))
                    #result = pandas2ri.rpy2py_dataframe(result)
                result = 2 ** result
                proteins = result.index.astype(int)
                result = result.to_numpy()
                for i, id in enumerate(idx):
                    self.array[proteins, id, fidx] = result[:, i]
        self.array[all_missing] = 0  # prevents imputing where all fractions are zero
        self._impute_fix_df()  # makessure values in the initial dataframe are similar to the ones in array


    def _check_distances(self):
        if self.distances is None:
            raise ValueError("Distances not calculated yet. Need to calculate distances first for this operation")

    def calc_distance_stats(self):
        """ Calculates the mean distance and variance of this distance inside the same treatment group.
        """
        self._check_distances()
        for treatment, idx in zip(self.treatment_levels, self.indices):
            indices = np.triu_indices(len(idx), 1)
            sub_distances = self.distances[:, idx][:, : , idx][:, indices[0], indices[1]]
            d_mean = np.nanmean(sub_distances, axis=-1)
            d_var = np.nanvar(sub_distances, axis=-1)
            self.df[f"{treatment} distance mean"] = d_mean
            self.df[f"{treatment} distance var"] = d_var





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
            raise ValueError(f"methhod: {method} is not supported. Must be one of {self.methods}")
        return distances

    def calc_distances(self, method: str = None):
        """Calculates between sample distances.
                
        Args:
            method (str): One of the values from `methods`. The method used for sample distance calculation.

        Raises:
            ValueError: If the method string is not supported or symmetric-kl-divergence is used without adding an
                epsilon to the protein intensities

        """
        if method is None:
            method = self.methods[0]
        array = self.norm_array
        array1, array2 = array[:, :, :, None], array[:, :, :, None].transpose(0, 3, 2, 1)
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
        warnings.warn("This function is deprecated use run_preprocessing instead", DeprecationWarning)
        self.run_preprocessing(
            method=method,
            kernel=kernel,
            eps=eps
        )

    def run_preprocessing(
            self, method: str = None,
            kernel: int = 0,
            impute: bool = False,
            impute_perc: float = 0.5,
            impute_nn: int = 10,
            impute_quantile: float = 0.95,
            eps: float = 0
    ):
        """Normalizes the array, imputes missing values if needed and calculates sample distances.

        Args:
            method (str): One of the values from `methods`. The method used for sample distance calculation.
            kernel (int): Averaging kernel size. This kernel is applied to the fractions.
            impute (bool): Wheter to impute missing values using KNN
            impute_perc (float): Maximum percentage of missing values per fraction and condition used for imputation.
            impute_nn (int): N Neighbors to account in data imputation.
            impute_quantile (int): Does only impute values for the quantile with the lowest mean distance between
                replicates. Others are assumed to be too noisy and data imputation might cause problems.
            eps (float): epsilon added to the intensities to overcome problems with zeros.
        """
        if method is None:
            method = self.methods[0]

        if impute:
            self.normalize_array_with_kernel()
            self.calc_distances(method)
            self._impute_via_rpy(impute_perc, impute_nn, impute_quantile)
        self.normalize_array_with_kernel(kernel, eps)

        self.calc_distances(method)
        self._unset_scores_and_pvalues()

    def determine_strongest_shift(self):
        """Determines the position of the strongest shift
        """
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
                f = self.fractions[i:-i] if i else self.fractions
                t = ((test == np.max(test, axis=-1, keepdims=True)) * f)
                positions = t.sum(axis=-1) / np.count_nonzero(t, axis=-1)
                positions = np.floor(positions).astype(int)
        # Get the middle occurrence index
        self.df["position strongest shift"] = positions
        self.df.loc[self.df["Mean Distance"].isna(), "position strongest shift"] = pd.NA

    def calc_mean_distance(self):
        """ Calculates the distance between the means of the two treatment groups.
        """
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


    def calc_distribution_features(self):
        """ Calculates features used in a bubble plot

        Sets the features in :attr:`current_embedding` that can be used to plot a bubble plot of the data

        """
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


    def remove_clusters(self):
        if "Cluster" in self.df:
            self.df = self.df.drop("Cluster", axis=1)
        self.cluster_features = None
        self.state.cluster_method = None
        self.state.cluster_args = None


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



    def _calc_permanova_f(self, indices_false, indices_true):
        assert len(indices_true) == len(indices_false), "PERMANOVA performs poorly for unbalanced study design"
        outer_group_distances = _get_outer_group_distances(indices_false, indices_true, self.distances)
        inner_group_distances = _get_innergroup_distances(indices_false, indices_true, self.distances)
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
        r = _calc_anosim(self.distances, self.indices[0], self.indices[1])
        r[self.df["Mean Distance"] == 0] = np.nan
        r[self.df["min replicates per group"] < self.min_replicates] = np.nan

        self.df["ANOSIM R"] = r.round(decimals=DECIMALS)
        self.state.anosim_r = True

    def _calc_global_anosim_distribution(self, nr_permutations: int, threads: int, seed: int = 0, ignore_zero_distances = True, callback=None):
        np.random.seed(seed)
        _split_point = len(self.indices[0])
        indices = np.concatenate((self.indices[0], self.indices[1]))
        calls = []
        if nr_permutations == -1:
            perms = it.permutations(indices)
            for shuffled in perms:
                calls.append((_calc_anosim, self.distances, shuffled[:_split_point], shuffled[_split_point:]))
        else:
            for _ in range(nr_permutations):
                shuffled = np.random.permutation(indices)
                calls.append((_calc_anosim, self.distances, shuffled[:_split_point], shuffled[_split_point:]))
        if threads > 1:
            with multiprocessing.Manager() as manager:
                counter = manager.Value('i', 0)
                lock = manager.Lock()
                total_calls = len(calls)
                if callback:
                    watcher_thread = threading.Thread(target=_progress_watcher,
                                                      args=(counter, total_calls, callback, lock))
                    watcher_thread.start()
                with multiprocessing.Pool(
                        threads,
                        initializer=_init_progress,
                        initargs=(counter, total_calls, lock)
                ) as pool:
                    result = pool.starmap(_calc_wrapper, calls)

                if callback:
                    watcher_thread.join()
        else:
            if callback:
                result = []
                m_len = len(calls)
                for idx, call in enumerate(calls):
                    perc = str(int((idx * 97) / m_len))
                    callback(perc)
                    result.append(_calc_anosim(*(call[1:])))
            else:
                result = [_calc_anosim(*(call[1:])) for call in calls]
        result = np.stack(result)
        result[:, self.df["min replicates per group"] < self.min_replicates] = np.nan
        if ignore_zero_distances:
            result[:, np.all(self.distances == 0, axis=(1, 2))] = np.nan
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
        if "Mean Distance" not in self.df.columns:
            self.calc_mean_distance()

        if "ANOSIM R" not in self.df.columns:
            self.calc_all_anosim_value()


        o_distribution = self._calc_global_anosim_distribution(permutations, threads, seed, callback=callback)
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

    def pca(self):
        """ Performs PCA on the normalized array.

        Results are stored in :attr:`df` as PC1 and PC2. explained variance is stored in :attr:`pca_var`.

        """
        p, r, f = self.norm_array.shape
        X = self.norm_array.reshape(p, r * f)
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]
        # Step 2: Perform PCA on valid rows
        pca = PCA(n_components=2)
        X_valid_pca = pca.fit_transform(X_valid)
        # Step 3: Create output array with NaNs
        X_pca = np.full((X.shape[0], X_valid_pca.shape[1]), np.nan)
        X_pca[valid_mask] = X_valid_pca
        self.df["PC1"] = X_pca[:, 0]
        self.df["PC2"] = X_pca[:, 1]
        self.pca_var = pca.explained_variance_ratio_


    def export_csv(self, file: str, sep: str = ","):
        """Exports the :attr:`extra_df` to a file.

        Args:
            file (str): Path to file where dataframe should be exported to.
            sep (str): seperator to use.

        """
        df = self.extra_df.drop(["id"], axis=1)
        df.to_csv(file, sep=sep, index=False)

    def to_jsons(self):
        """ encodes this object as a JSON string

        Returns:
            str: JSON string representation of this object

        """
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
        """ Creates class instance from JSON string.

        Args:
            json_string: string representation of the RAPDORData

        Returns:
            RAPDORData: the RAPDORData stored in the string

        """
        json_obj = json.loads(json_string)
        data = cls._from_dict(json_obj)
        return data

    @classmethod
    def from_file(cls, json_file):
        """ Creates a class instance from a JSON file.

        Returns:
            RAPDORData: the RAPDORData stored in the file.
        """
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
    """ The JSON encoder used to create the JSON representation of a RAPDORData instance.

    """
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

def _calc_anosim(distances, indices_false, indices_true, ignore_nan: bool = True):
    outer_group_distances = _get_outer_group_distances(indices_false, indices_true, distances,)
    inner_group_distances = _get_innergroup_distances(indices_false, indices_true, distances,)
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
    return r


def _get_innergroup_distances(indices_false, indices_true, distances):
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


def _get_outer_group_distances(indices_false, indices_true, distances):
    n_genes = distances.shape[0]
    mg1, mg2 = np.meshgrid(indices_true, indices_false)
    e = np.ones((n_genes, len(indices_false), len(indices_true)))
    e = e * np.arange(0, n_genes)[:, None, None]
    e = e[np.newaxis, :]
    e = e.astype(int)
    mg = np.stack((mg1, mg2))

    mg = np.repeat(mg[:, np.newaxis, :, :], n_genes, axis=1)

    idx = np.concatenate((e, mg))
    distances = distances.flat[np.ravel_multi_index(idx, distances.shape)]
    distances = distances.reshape((n_genes, len(indices_true) * len(indices_false)))
    return distances



def _init_progress(counter_, total_, lock_):
    global counter, total, lock
    counter = counter_
    total = total_
    lock = lock_

def _calc_wrapper(fct, *args):
    res = fct(*args)  # make sure this is a staticmethod
    with lock:
        counter.value += 1
    return res

def _progress_watcher(counter, total, callback, lock):
    last_reported = -1
    while True:
        with lock:
            current = counter.value
        perc = int((current * 97) / total)
        if perc != last_reported:
            callback(str(perc))
            last_reported = perc
        if current >= total:
            break
        time.sleep(0.2)


if __name__ == '__main__':
    from RAPDOR.plots import COLOR_SCHEMES
    from RAPDOR.plots import plot_distance_stats
    f = "/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/RAPDORAnalysis/sanitized_df.tsv"
    f = "/home/rabsch/PythonProjects/synRDPMSpec/Data/synIntensitiesIBAQ.tsv"
    d = "/home/rabsch/PythonProjects/synRDPMSpec/Data/synDesignIBAQ.tsv"
    #f  = "../testData/testFile.tsv"
    #d  = "../testData/testDesign.tsv"
    df = pd.read_csv(f, sep="\t")
    dolphin = COLOR_SCHEMES["Dolphin"]

    design = pd.read_csv(d, sep="\t")
    rapdor = RAPDORData(df, design, logbase=2)
    #rapdor._impute(2)
    rapdor.run_preprocessing("Jensen-Shannon-Distance", kernel=3, impute=True, impute_quantile=0.95)
    rapdor.calc_distances()
    rapdor.calc_anosim_p_value(-1, threads=10, mode="global")
    print(np.nanmax(rapdor.df[rapdor.df["ANOSIM R"] >=0.5]["global ANOSIM adj p-Value"]))
    exit()
    #rapdor.calc_distance_stats()
    fig = plot_distance_stats(rapdor, dolphin)
    fig.show()
    rapdor.to_json("testfile.json")
    from RAPDOR.plots import plot_sample_correlation, plot_ep_vs_ep, plot_sample_pca
    #fig = plot_sample_correlation(rapdor, method="spearman")
    fig = plot_sample_correlation(rapdor, summarize_fractions=True, method="spearman", colors=[dolphin[0], "white", dolphin[1]])
    fig.show()
    fig = plot_sample_pca(rapdor, summarize_fractions=True, ntop=0.2, plot_dims=(1, 2), colors=dolphin)
    fig.show()


    #fig.show()
    exit()
