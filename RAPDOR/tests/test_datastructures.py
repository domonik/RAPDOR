import dash
import pytest
import pandas as pd
import os
from RAPDOR.datastructures import RAPDORData
import numpy as np

np.random.seed(42)


TESTFILE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(TESTFILE_DIR, "testData")


@pytest.fixture(scope="session")
def intensities():
    file = os.path.join(TESTDATA_DIR, "testFile.tsv")
    df = pd.read_csv(file, sep="\t")
    df.index = df.index.astype(str)
    return df


@pytest.fixture(scope="session")
def design():
    file = os.path.join(TESTDATA_DIR, "testDesign.tsv")
    df = pd.read_csv(file, sep="\t")
    return df


@pytest.fixture()
def rapdordata(intensities, design):
    rapdordata = RAPDORData(intensities, design, 2)
    return rapdordata

@pytest.fixture()
def norm_rapdordata(rapdordata):
    rapdordata.normalize_and_get_distances("Jensen-Shannon-Distance", 3, eps=0)
    return rapdordata

@pytest.fixture()
def scored_rapdordata(norm_rapdordata):
    norm_rapdordata.calc_all_scores()
    return norm_rapdordata


@pytest.fixture(scope="session")
def multi_design(design):
    designs = []
    for i in range(3):
        cdesign = design.copy()
        cdesign["Name"] += f".{i}"
        cdesign["Replicate"] += f".{i}"
        designs.append(cdesign)
    new_design = pd.concat(designs, ignore_index=True)
    return new_design


@pytest.fixture(scope="session")
def multi_intensities(intensities):
    result = None
    for i in range(3):
        cdf = intensities.copy()
        cdf.columns = [cdf.columns[0]] + [f"{col}.{i}" for col in cdf.columns[1:]]
        if i == 0:
            result = cdf
        else:
            result = pd.merge(result, cdf, left_index=True, right_index=True)
    return result


def drop_replicates(design, rnase_rep, ctrl_rep):
    if rnase_rep > 0:
        rnase = design[design["Treatment"] == "RNase"].groupby("Replicate").apply(lambda x: list(x.index), include_groups=False).sample(n=rnase_rep).sum()
    else:
        rnase = []
    if ctrl_rep > 0:
        ctrl = design[design["Treatment"] == "Control"].groupby("Replicate").apply(lambda x: list(x.index), include_groups=False).sample(n=ctrl_rep).sum()
    else:
        ctrl = []
    rnase = design.loc[rnase]
    ctrl = design.loc[ctrl]
    new_design = pd.concat((rnase, ctrl), ignore_index=True)
    return new_design






@pytest.mark.parametrize(
    "rnase_rep,ctrl_rep",
    [
        (4, 7),
        (9, 9),
    ]
)
def test_multi_design(rnase_rep, ctrl_rep, multi_design, multi_intensities):
    multi_design = drop_replicates(multi_design, rnase_rep, ctrl_rep)

    rapdordata = RAPDORData(multi_intensities, multi_design, 2)
    rapdordata.normalize_and_get_distances("Jensen-Shannon-Distance", 3)
    rapdordata.calc_all_scores()
    s = rapdordata.to_jsons()
    loaded_data = RAPDORData.from_json(s)
    assert loaded_data == rapdordata


@pytest.mark.parametrize(
    "rnase_rep,ctrl_rep",
    [
        (0, 3),
        (6, 0),
    ]
)
def test_wrong_treatment_levels(rnase_rep, ctrl_rep, multi_design, multi_intensities):
    multi_design = drop_replicates(multi_design, rnase_rep, ctrl_rep)
    with pytest.raises(ValueError):
        _ = RAPDORData(multi_intensities, multi_design, 2)


def test_treatment_names(intensities, design):
    design["Treatment"] = ["Z" + t if t == "Control" else t for t in design["Treatment"]]
    rapdordata = RAPDORData(intensities, design, 2, control="ZControl")
    rapdordata.normalize_array_with_kernel(3)
    rapdordata.calc_distances("Jensen-Shannon-Distance")
    rapdordata.calc_all_scores()
    rapdordata.rank_table(['Mean Distance', "ANOSIM R"], [False, False])

    s = rapdordata.to_jsons()
    loaded_data = RAPDORData.from_json(s)
    assert loaded_data == rapdordata


def test_different_columns(intensities, design):
    intensities = intensities[["id"] + [col for col in intensities.columns if "LFQ" in col]]
    rapdordata = RAPDORData(intensities, design, 2)
    rapdordata.normalize_array_with_kernel(3)
    rapdordata.calc_distances("Jensen-Shannon-Distance")
    rapdordata.calc_all_scores()
    rapdordata.rank_table(['Mean Distance', "ANOSIM R"], [False, False])

    s = rapdordata.to_jsons()
    loaded_data = RAPDORData.from_json(s)
    assert loaded_data == rapdordata




@pytest.mark.parametrize(
    "normalize,kernel_size,distance,permanova,nr_samples",
    [
        (True, 3, "Jensen-Shannon-Distance", True, 10),
        (False, None, None, False, None)
    ]
)
def test_serialization(normalize, kernel_size, distance, permanova, nr_samples, rapdordata):
    if normalize:
        rapdordata.normalize_array_with_kernel(kernel_size)
    if distance:
        rapdordata.calc_distances(distance)
    if permanova:
        rapdordata.calc_all_scores()
        rapdordata.rank_table(['Mean Distance', "ANOSIM R"], [False, False])


        rapdordata.calc_permanova_p_value(10, threads=1)
    s = rapdordata.to_jsons()
    loaded_data = RAPDORData.from_json(s)
    assert loaded_data == rapdordata

# @pytest.mark.parametrize(
#     "cluster_method,feature_kernel_size",
#     [
#         ("HDBSCAN", 3),
#
#     ]
# )
# def test_cluster_serialization(cluster_method, feature_kernel_size, scored_rapdordata):
#     scored_rapdordata.calc_cluster_features(kernel_range=feature_kernel_size)
#     scored_rapdordata.cluster_data(method=cluster_method)
#     s = scored_rapdordata.to_jsons()
#     loaded_data = RAPDORData.from_json(s)
#     assert loaded_data == scored_rapdordata
#
#
# @pytest.mark.parametrize(
#     "method",
#     ["T-SNE", "PCA", "UMAP"]
# )
# @pytest.mark.parametrize(
#     "dimension",
#     [2, 3]
# )
# def test_reduced_dim_serialization(method, dimension, scored_rapdordata):
#     scored_rapdordata.calc_cluster_features(kernel_range=3)
#     scored_rapdordata.reduce_dim(scored_rapdordata.cluster_features, dimension, method)
#     s = scored_rapdordata.to_jsons()
#     loaded_data = RAPDORData.from_json(s)
#     assert loaded_data == scored_rapdordata


def test_to_json(rapdordata):
    rapdordata.to_jsons()
