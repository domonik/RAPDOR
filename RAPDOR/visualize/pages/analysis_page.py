import dash
from dash import html, dcc
from RAPDOR.visualize.distributionAndHeatmap import distribution_panel, distance_heatmap_box
from RAPDOR.visualize.dataTable import _get_table
from RAPDOR.visualize.clusterAndSettings import _get_cluster_panel, selector_box
from RAPDOR.visualize.callbacks.mainCallbacks import *
from RAPDOR.visualize.callbacks.plotCallbacks import * # Don´t delete that. It is needed.
from RAPDOR.visualize.callbacks.tableCallbacks import *
from RAPDOR.visualize.callbacks.modalCallbacks import *
from RAPDOR.visualize.callbacks.colorCallbacks import *
import os
from RAPDOR.visualize.modals import (
    _modal_image_download,
    _modal_hdbscan_cluster_settings,
    _modal_dbscan_cluster_settings,
    _modal_kmeans_cluster_settings,
    _modal_cluster_image_download
)
from RAPDOR.visualize.colorSelection import _color_theme_modal, _modal_color_selection
from RAPDOR.visualize import DISABLED



dash.register_page(__name__, path='/analysis')

layout = html.Div(
    [
        html.Div(
            distribution_panel("None"),
            className="row px-2 justify-content-center align-items-center"

        ),
        html.Div(id="test-div", style={"display": "none", "height": "0%"}),
        dcc.Tabs(
            [
                dcc.Tab(
                    html.Div(
                        _get_table(rapdordata=None),
                        className="row px-2 justify-content-center align-items-center",
                        id="protein-table"
                    ),
                    label="Table", className="custom-tab", selected_className='custom-tab--selected', id="table-tab"
                ),
                dcc.Tab(
                    html.Div(
                        _get_cluster_panel(DISABLED),
                        className="row px-2 justify-content-center align-items-center"

                    ), label="Bubble Plot", className="custom-tab", selected_className='custom-tab--selected',
                    id="dim-red-tab"
                )
            ],
            id="analysis-tabs",
            parent_className='custom-tabs',
            className='custom-tabs-container',

        ),

        html.Div(
            [distance_heatmap_box(), selector_box(DISABLED)],
            className="row px-2 row-eq-height justify-content-center"
        ),
        _modal_image_download(),
        _modal_cluster_image_download(),
        _modal_color_selection("primary"),
        _modal_color_selection("secondary"),
        _modal_hdbscan_cluster_settings(),
        _modal_dbscan_cluster_settings(),
        _modal_kmeans_cluster_settings(),
        _color_theme_modal(),

        html.Div(id="recomputation"),
        html.Button("refresh", className="btn-primary", id="refresh-btn", style={"display": "none"})
    ]
)
