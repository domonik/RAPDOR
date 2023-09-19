import dash
from dash import html, dcc
from RDPMSpecIdentifier.visualize.distributionAndHeatmap import distribution_panel, distance_heatmap_box
from RDPMSpecIdentifier.visualize.dataTable import _get_table
from RDPMSpecIdentifier.visualize.clusterAndSettings import _get_cluster_panel, selector_box
from RDPMSpecIdentifier.visualize.callbacks.mainCallbacks import *
from RDPMSpecIdentifier.visualize.callbacks.plotCallbacks import * # Don´t delete that. It is needed.
from RDPMSpecIdentifier.visualize.callbacks.tableCallbacks import *
from RDPMSpecIdentifier.visualize.callbacks.modalCallbacks import *
import os
from RDPMSpecIdentifier.visualize.modals import (
    _modal_image_download,
    _modal_color_selection,
    _modal_hdbscan_cluster_settings,
    _modal_dbscan_cluster_settings,
    _modal_kmeans_cluster_settings,
    _modal_cluster_image_download
)
from RDPMSpecIdentifier.visualize import DISABLED



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
                        _get_table(rdpmsdata=None),
                        className="row px-2 justify-content-center align-items-center",
                        id="protein-table"
                    ),
                    label="Table", className="custom-tab", selected_className='custom-tab--selected'
                ),
                dcc.Tab(
                    html.Div(
                        _get_cluster_panel(DISABLED),
                        className="row px-2 justify-content-center align-items-center"

                    ), label="Clustering", className="custom-tab", selected_className='custom-tab--selected'
                )
            ],
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
        html.Div(id="recomputation"),
        html.Button("refresh", className="btn-primary", id="refresh-btn", style={"display": "none"})
    ]
)
