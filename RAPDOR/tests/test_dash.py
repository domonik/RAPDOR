from RAPDOR.visualize.appDefinition import app
from RAPDOR.datastructures import RAPDORData
from RAPDOR.visualize.pages.upload_page import upload_from_csv
import base64
from dash import dcc
import dash
import os
import pytest

from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict



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

    def run_callback():
        context_value.set(AttributeDict(**{"triggered_inputs": [{"prop_id": "upload-csv-btn.n_clicks"}]}))
        return upload_from_csv(btn, None, uid, sep, base64intensities, base64design, logbase)

    ctx = copy_context()
    rapdordata, redirect, alert, *state3 = ctx.run(run_callback)
    assert redirect == "analysis"
    assert alert == []
    assert isinstance(rapdordata.value, RAPDORData)
    for state in state3:
        assert state is None
    #assert state2 is None
    #assert state is None
