
import dash
from dash import html
import base64
from dash import dcc, ctx
from dash_extensions.enrich import callback, Input, Output, Serverside, State
from RAPDOR.datastructures import RAPDORData
import logging
import dash_bootstrap_components as dbc
import pandas as pd
from io import StringIO
from dash.exceptions import PreventUpdate
import os
from RAPDOR.visualize import DISABLED, DISPLAY, WELCOME_FILES
import re
import ast
from xml.etree import ElementTree

logger = logging.getLogger(__name__)
if DISPLAY:
    dash.register_page(__name__, path='/')

    def replace_relative_links(markdown_text, markdown_path, python_path):
        # Extract directory path of the markdown file
        markdown_dir = os.path.dirname(markdown_path)

        # Extract directory path of the python file
        python_dir = os.path.dirname(python_path)

        # Replace relative links in img tags
        def replace_link(match):
            link = match.group(1)
            # Make the link path relative to the Python file
            img_path = os.path.join(markdown_dir, link)

            # Determine the file extension
            ext = os.path.splitext(img_path)[1].lower()

            # Read image file
            with open(img_path, 'rb') as f:
                img_data = f.read()

            # Encode based on image format
            if ext == '.svg':
                img_data_encoded = base64.b64encode(img_data).decode()
                img_src = f'data:image/svg+xml;base64,{img_data_encoded}'
            elif ext in ('.png', '.jpg', '.jpeg'):
                img_data_encoded = base64.b64encode(img_data).decode()
                img_src = f'data:image/{ext[1:]};base64,{img_data_encoded}'
            else:
                raise ValueError("Unsupported image format")

            return f'src="{img_src}"'

        # Regular expression to match HTML img tags and extract src attribute
        img_regex = r'src="([^"]+)"'

        # Replace links in the markdown text
        new_markdown_text = re.sub(img_regex, replace_link, markdown_text)

        return new_markdown_text

    WELCOME_TEXTS = []

    for file in WELCOME_FILES:
        with open(file) as handle:
            text = handle.read()
        md_file = os.path.abspath(file)
        assert os.path.exists(md_file)
        text = replace_relative_links(text, md_file, os.path.abspath(__file__))
        WELCOME_TEXTS.append(text)

    def welcome_layout():
        welcome = [

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    dcc.Markdown(text, dangerously_allow_html=True,),
                                ]
                            )
                        ]
                        , className="databox p-2", style={"font-size": "20px"})

                ],
                className="col-12 col-lg-6 p-2 equal-height-column" if len(WELCOME_TEXTS) > 2 else "col-12 col-lg-6 p-2"
            ) for text in WELCOME_TEXTS
        ]

        return welcome

    layout = html.Div(
        welcome_layout(), className="row p-1 justify-content-between" if len(WELCOME_TEXTS) > 1 else "row p-1 justify-content-around"
    )
