import dash
import pytest
import pandas as pd
import os
from RDPMSpecIdentifier.datastructures import RDPMSpecData


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


@pytest.fixture(scope="session")
def rdpmpecdata(intensities, design):
    rdpmsdata = RDPMSpecData(intensities, design, 2)
    return rdpmsdata


@pytest.fixture(scope="session")
def norm_rdpmspecdata(rdpmpecdata):
    rdpmpecdata.normalize_and_get_distances("Jensen-Shannon-Distance", 3, eps=0)
    return rdpmpecdata


@pytest.fixture(scope="session")
def scored_rdpmspecdata(norm_rdpmspecdata):
    norm_rdpmspecdata.calc_all_scores()
    return norm_rdpmspecdata



@pytest.mark.parametrize(
    "normalize,kernel_size,distance,permanova,nr_samples, distance_cutoff",
    [
        (True, 3, "Jensen-Shannon-Distance", True, 10, 0.1),
        (True, 3, "Jensen-Shannon-Distance", True, 10, 0),
        (False, None, None, False, None, None)
    ]
)
def test_serialization(normalize, kernel_size, distance, permanova, nr_samples, distance_cutoff, rdpmpecdata):
    if normalize:
        rdpmpecdata.normalize_array_with_kernel(kernel_size)
    if distance:
        rdpmpecdata.calc_distances(distance)
    if permanova:
        rdpmpecdata.calc_all_scores()
        rdpmpecdata.calc_permanova_p_value(10, threads=1, distance_cutoff=distance_cutoff)
    s = rdpmpecdata.to_jsons()
    loaded_data = RDPMSpecData.from_json(s)
    assert loaded_data == rdpmpecdata

@pytest.mark.parametrize(
    "cluster_method,feature_kernel_size",
    [
        ("HDBSCAN", 3),

    ]
)
def test_cluster_serialization(cluster_method, feature_kernel_size, scored_rdpmspecdata):
    scored_rdpmspecdata.calc_cluster_features(kernel_range=feature_kernel_size)
    scored_rdpmspecdata.cluster_data(method=cluster_method)
    s = scored_rdpmspecdata.to_jsons()
    loaded_data = RDPMSpecData.from_json(s)
    assert loaded_data == scored_rdpmspecdata


@pytest.mark.parametrize(
    "method",
    ["T-SNE", "PCA", "UMAP"]
)
@pytest.mark.parametrize(
    "dimension",
    [2, 3]
)
def test_reduced_dim_serialization(method, dimension, scored_rdpmspecdata):
    scored_rdpmspecdata.calc_cluster_features(kernel_range=3)
    scored_rdpmspecdata.reduce_dim(scored_rdpmspecdata.cluster_features, dimension, method)
    s = scored_rdpmspecdata.to_jsons()
    loaded_data = RDPMSpecData.from_json(s)
    assert loaded_data == scored_rdpmspecdata


def test_to_json(rdpmpecdata):
    rdpmpecdata.to_jsons()
