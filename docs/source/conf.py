# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

FPATH = os.path.abspath(__file__)
__RDPMSPECPATH = os.path.abspath(os.path.join(FPATH, "../../"))
__RDPMSPECPATH = os.path.abspath("../../")
sys.path.insert(1, __RDPMSPECPATH)
from RDPMSpecIdentifier import _version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RDPMSpecIdentifier'
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
release = __version__

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
    'sphinxarg.ext',
    'sphinx_design',
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
        "image_light": "RDPMSpecIdentifier_light.svg",
        "image_dark": "RDPMSpecIdentifier_dark.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/domonik/RDPMSpecIdentifier",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },

    ]
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
    "scipy",
    "sklearn",
    "skbio",
    "statsmodels",
    "dash_loading_spinners"
]