from RDPMSpecIdentifier.visualize.appDefinition import app
from RDPMSpecIdentifier.datastructures import RDPMSpecData
from RDPMSpecIdentifier.visualize.pages.upload_page import upload_from_csv
import base64
from dash import dcc
import dash
import os
import pytest



TESTFILE_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(TESTFILE_DIR, "testData")


@pytest.fixture(scope="session")
def base64intensities():
    file = os.path.join(TESTDATA_DIR, "testFile.tsv")
    return dcc.send_file(file)


@pytest.fixture(scope="session")
def base64design():
    file = os.path.join(TESTDATA_DIR, "testDesign.tsv")
    return dcc.send_file(file)


def test_csv_upload(base64design, base64intensities):
    btn = 1
    uid = 1
    sep = "\t"
    logbase = 2
    base64design = "base64," + base64design["content"]
    base64intensities = "base64," + base64intensities["content"]
    rdpmsdata, redirect, alert = upload_from_csv(btn, uid, sep, base64intensities, base64design, logbase)
    assert redirect == "analysis"
    assert alert == []
    assert isinstance(rdpmsdata.value, RDPMSpecData)
