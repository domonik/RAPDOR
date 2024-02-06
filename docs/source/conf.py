# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

FPATH = os.path.abspath(__file__)
__RAPDORPECPATH = os.path.abspath(os.path.join(FPATH, "../../"))
__RAPDORPECPATH = os.path.abspath("../../")
assert os.path.exists(__RAPDORPECPATH)
from unittest.mock import Mock
sys.path.insert(1, __RAPDORPECPATH)
from RAPDOR import _version

IMG_LIGHT = os.path.join(__RAPDORPECPATH, "RAPDOR", "visualize", "assets", "RAPDOR_light.svg")
IMG_DARK = os.path.join(__RAPDORPECPATH, "RAPDOR", "visualize", "assets", "RAPDOR.svg")


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'RAPDOR'
copyright = '2023, Domonik'
author = 'Domonik'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# The full version, including alpha/beta/rc tags
__version__ = _version.get_versions()["version"]
wd_v = __version__.split("+")
if len(wd_v) > 1:
    if wd_v[1][0] == "0":
        __version__ = wd_v[0]
__version__ = __version__.replace(".dirty", "")
release = __version__

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    'sphinxarg.ext',
    'sphinx_design',
    'sphinx_copybutton',
    "sphinx_multiversion",
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # Toc options
    "collapse_navigation": True,
    "navigation_depth": 4,
    "show_prev_next": False,
    "logo": {
        "image_light": IMG_LIGHT,
        "image_dark": IMG_DARK,
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["versions", "navbar-nav"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/domonik/RAPDOR",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },

    ],
    "pygment_light_style": "tango",
    "pygment_dark_style": "dracula"

}
mathjax3_config = {'chtml': {'displayAlign': 'left'}}
html_static_path = ['_static']

html_css_files = [
    'args.css',
]
html_favicon = '_static/favicon.ico'


autosummary_mock_imports = [
    "plotly",
    "numpy",
    "numpy.core.multiarray",
    "pandas",
    "dash_bootstrap_components",
    "dash",
    "plotly.colors.qualitative.Light24",
    "plotly.colors.qualitative.Dark24",
    "plotly.colors.qualitative.G10",
    "plotly.colors.qualitative.Alphabet",
    "plotly.colors.qualitative",
    "scipy",
    "sklearn",
    "RAPDOR.visualize",
    "RAPDOR.qtInterface",
    "RAPDOR.executables",
    "statsmodels",
    "dash_loading_spinners",
    "dash_daq",
    "umap"
]

smv_tag_whitelist = r'^.*$'
smv_branch_whitelist = r'^(fo)$'

sys.modules["RAPDOR.visualize.runApp"] = Mock()