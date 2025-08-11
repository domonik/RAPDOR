import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Iterable, Tuple, List, Any, Dict, Union
from plotly.colors import qualitative
from RAPDOR.datastructures import RAPDORData
import plotly.io as pio
import copy
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from plotly.validator_cache import ValidatorCache

SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
SYMBOLS = SymbolValidator.values[2::12]

SymbolValidator3D = ValidatorCache.get_validator("scatter3d.marker", "symbol")
SYMBOLS3D = SymbolValidator3D.values

DEFAULT_COLORS = {"primary": "rgb(138, 255, 172)", "secondary": "rgb(255, 138, 221)"}

COLOR_SCHEMES = {
    "Flamingo": (DEFAULT_COLORS["primary"], DEFAULT_COLORS["secondary"]),
    "Viking": ("rgb(79, 38, 131)", "rgb(255, 198, 47)"),
    "Dolphin": ("rgb(0, 142, 151)", "rgb(252, 76, 2)"),
    "Cardinal": ("rgb(151,35,63)", "rgb(0,0,0)"),
    "Falcon": ("rgb(167, 25, 48)", "rgb(0, 0, 0)"),
    "Raven": ("rgb(26, 25, 95)", "rgb(0, 0, 0)"),
    "Bill": ("rgb(0, 51, 141)", "rgb(198, 12, 48)"),
    "Panther": ("rgb(0, 133, 202)", "rgb(16, 24, 32)"),
    "Bear": ("rgb(111, 22, 42)", "rgb(12, 35, 64)"),
    "Bengal": ("rgb(251, 79, 20)", "rgb(0, 0, 0)"),
    "Brown": ("rgb(49, 29, 0)", "rgb(255, 60, 0)"),
    "Cowboy": ("rgb(0, 34, 68)", "rgb(255, 255, 255)"),
    "Bronco": ("rgb(251, 79, 20)", "rgb(0, 34, 68)"),
    "Lion": ("rgb(0, 118, 182)", "rgb(176, 183, 188)"),
    "Packer": ("rgb(24, 48, 40)", "rgb(255, 184, 28)"),
    "Texan": ("rgb(3, 32, 47)", "rgb(167, 25, 48)"),
    "Colt": ("rgb(0, 44, 95)", "rgb(162, 170, 173)"),
    "Jaguar": ("rgb(215, 162, 42)", "rgb(0, 103, 120)"),
    "Chief": ("rgb(227, 24, 55)", "rgb(255, 184, 28)"),
    "Charger": ("rgb(0, 128, 198)", "rgb(255, 194, 14)"),
    "Ram": ("rgb(0, 53, 148)", "rgb(255, 163, 0)"),
    "Patriot": ("rgb(0, 34, 68)", "rgb(198, 12, 48)"),
    "Saint": ("rgb(211, 188, 141)", "rgb(16, 24, 31)"),
    "Giant": ("rgb(1, 35, 82)", "rgb(163, 13, 45)"),
    "Jet": ("rgb(18, 87, 64)", "rgb(255, 255, 255)"),
    "Raider": ("rgb(0, 0, 0)", "rgb(165, 172, 175)"),
    "Eagle": ("rgb(0, 76, 84)", "rgb(165, 172, 175)"),
    "Steeler": ("rgb(255, 182, 18)", "rgb(16, 24, 32)"),
    "49": ("rgb(170, 0, 0)", "rgb(173, 153, 93)"),
    "Seahawk": ("rgb(0, 34, 68)", "rgb(105, 190, 40)"),
    "Buccaneer": ("rgb(213, 10, 10)", "rgb(255, 121, 0)"),
    "Titan": ("rgb(75, 146, 219)", "rgb(200, 16, 46)"),
    "Commander": ("rgb(90, 20, 20)", "rgb(255, 182, 18)"),
}

COLORS = list(qualitative.Alphabet) + list(qualitative.Light24) + list(qualitative.Dark24) + list(qualitative.G10)

DEFAULT_TEMPLATE = copy.deepcopy(pio.templates["plotly_white"])

DEFAULT_TEMPLATE.update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "rgba(255,255,255,1)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": dict(color="black"),
            "xaxis": dict(linecolor="black", showline=True, mirror=True),
            "yaxis": dict(linecolor="black", showline=True, mirror=True),
            "coloraxis": dict(colorbar=dict(outlinewidth=1, outlinecolor="black", tickfont=dict(color="black"))),
            "legend": dict(bgcolor="rgba(0,0,0,0)")
        }
    }
)

DEFAULT_TEMPLATE_DARK = copy.deepcopy(DEFAULT_TEMPLATE)

DEFAULT_TEMPLATE_DARK.update(
    {
        "layout": {
            # e.g. you want to change the background to transparent
            "paper_bgcolor": "#181818",
            "plot_bgcolor": " rgba(0,0,0,0)",
            "font": dict(color="white"),
            "coloraxis": dict(colorbar=dict(outlinewidth=0, tickfont=dict(color="white"))),
            "xaxis": dict(linecolor="white", showline=True),
            "yaxis": dict(linecolor="white", showline=True),

        }
    }
)


def plot_protein_pca(rapdordata, highlight = None, hovername: str = None, cutoff_range = None, cutoff_type = None, colors: Iterable = COLOR_SCHEMES["Flamingo"]):
    """
        Plots a 2D PCA (Principal Component Analysis) scatter plot for protein data.

        Args:
            rapdordata:
                An object containing protein PCA data. Must have the attributes:
                - `df`: A pandas DataFrame with PCA coordinates and metadata.
                - `pca_var`: A sequence with explained variances for PC1 and PC2.
            highlight (iterable, optional):
                A list or set of RAPDOR IDs to highlight in the plot.
                Highlighted points will be styled differently.
            hovername (str, optional):
                Name of the column in `rapdordata.df` to append to hover text.
            cutoff_range (tuple, optional):
                A tuple (min, max) specifying a numeric filter range.
                Used to subset the data based on values in the `cutoff_type` column.
            cutoff_type (str, optional):
                Name of the column in `rapdordata.df` used for filtering via `cutoff_range`.
                Must be provided if `cutoff_range` is set.
            colors (Iterable, optional):
                An iterable of two color values (e.g., hex strings).
                First color is used for non-highlighted points, second for highlighted points.
                Defaults to `COLOR_SCHEMES["Flamingo"]`.

        Returns:
            plotly.graph_objects.Figure:
                A Plotly scatter plot figure showing the PCA projection.

        Raises:
            AssertionError: If `cutoff_range` is provided without `cutoff_type`.

        Example:
            fig = plot_protein_pca(rapdordata, highlight=["P12345"], hovername="gene_name", cutoff_range=(0, 1), cutoff_type="q_value")
            fig.show()

    """
    df = rapdordata.df
    if highlight is not None:
        df["highlight"] = rapdordata.df["RAPDORid"].isin(highlight)
    else:
        df["highlight"] = False

    if cutoff_range is not None:
        assert cutoff_type is not None
        df = df[(df[cutoff_type] <= cutoff_range[1]) & (df[cutoff_type] >= cutoff_range[0])]

    hovertext = rapdordata.df.index.astype(str) + ": " + rapdordata.df["RAPDORid"].astype(str)
    if hovername is not None:
        hovertext = hovertext + "<br>" + rapdordata.df[hovername].astype(str)
    df["hovertext"] = hovertext
    fig = px.scatter(
        df,
        x="PC1",
        y=f"PC2",
        color="highlight",  # Color by highlight
        hover_name="hovertext",  # Show gene name on hover
        symbol="highlight",
        color_discrete_map={
            True: colors[1],  # Highlighted genes
            False: colors[0]  # All other genes
        },
    )
    fig.update_xaxes(title=f"PC1 ({rapdordata.pca_var[0] * 100:.2f}%)")
    fig.update_yaxes(title=f"PC2 ({rapdordata.pca_var[1] * 100:.2f}%)")
    return fig



def _color_to_calpha(color: str, alpha: float = 0.2):
    color = color.split("(")[-1].split(")")[0]
    return f"rgba({color}, {alpha})"


def _plot_pca(components, labels, to_plot: tuple = (0, 1, 2)):
    fig = go.Figure()
    x, y, z = to_plot

    for idx in range(components.shape[0]):
        fig.add_trace(
            go.Scatter3d(
                x=[components[idx, x]],
                y=[components[idx, y]],
                z=[components[idx, z]],
                mode="markers",
                marker=dict(color=DEFAULT_COLORS[labels[idx][1]]),
                name=labels[idx][0]
            )
        )
    return fig


def empty_figure(annotation: str = None, font_color: str = None):
    """
        Creates an empty Plotly figure with optional centered annotation text.

        This is useful as a placeholder figure in dashboards or when no data is available
        to display.

        Args:
            annotation (str, optional):
                A string to display as a centered annotation in the figure.
                If `None`, no annotation is shown.
            font_color (str, optional):
                Font color to use for the annotation and layout text (e.g., "#333333").
                If `None`, default color is used.

        Returns:
            plotly.graph_objects.Figure:
                An empty Plotly figure with customized layout and optional annotation.

        Example:
            fig = empty_figure(annotation="No data available", font_color="#555")
            fig.show()

    """
    fig = go.Figure()
    fig.update_yaxes(showticklabels=False, showgrid=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_layout(
        margin={"t": 0, "b": 0, "r": 50},
        font=dict(
            size=16,
        ),
        yaxis=dict(zeroline=False),
        xaxis=dict(zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",

    )
    if annotation is not None:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            text=annotation,
            showarrow=False,
            font=(dict(size=28))
        )
    fig.layout.template = "plotly_white"
    fig.update_layout(
        font=dict(color=font_color),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),

    )
    return fig


def plot_replicate_distribution(
        subdata: np.ndarray,
        design: pd.DataFrame,
        offset: int = 0,
        colors: Iterable[str] = None,
        yname: str = "rel. protein amount"
):
    """Plots the distribution of protein for each replicate

    Args:
        subdata (np.ndarray): an array of shape :code:`num samples x num_fractions`. Rows need to add up to one
        design (pd.Dataframe): the design dataframe to distinguish the groups from the samples dimension
        offset (int): adds this offset to the fractions at the x-axis range
        colors (Iterable[str]): An iterable of color strings to use for plotting

    Returns: go.Figure

        A plotly figure containing a scatter-line per replicate.

    """
    if colors is None:
        colors = DEFAULT_COLORS
    indices = design.groupby("Treatment", group_keys=True).apply(lambda x: list(x.index))
    if offset:
        x = list(range(offset + 1, subdata.shape[1] + offset + 1))
    else:
        x = list(range(subdata.shape[1]))
    fig = go.Figure()
    names = []
    values = []
    j_val = max([len(name) for name in indices.keys()])
    for eidx, (name, idx) in enumerate(indices.items()):
        name = f"{name}".ljust(j_val, " ")
        legend = f"legend{eidx + 1}"
        names.append(name)
        for row_idx in idx:
            rep = design.iloc[row_idx]["Replicate"]
            values.append(
                go.Scatter(
                    x=x,
                    y=subdata[row_idx],
                    marker=dict(color=colors[eidx]),
                    name=f"Replicate {rep}",
                    legend=legend,
                    line=dict(width=3)
                )
            )
    fig.add_traces(values)
    fig = _update_distribution_layout(fig, names, x, offset, yname=yname)
    return fig


def plot_protein_distributions(rapdorids, rapdordata: RAPDORData, colors, title_col: str = "RAPDORid",
                               mode: str = "line", plot_type: str = "normalized", zoom_fractions: int = 2, **kwargs):
    """Plots a figure containing distributions of proteins using mean, median, min and max values of replicates

        Args:
            rapdorids (List[any]): RAPDORids that should be plotted
            rapdordata (RAPDORData): a RAPDORData object containing the IDs from rapdorids
            colors (Iterable[str]): An iterable of color strings to use for plotting
            title_col (str): Name of a column that is present of the dataframe in rapdordata. Will add this column
                as a subtitle in the plot (Default: RAPDORid)
            mode (str): One of line or bar. Will result in a line plot or a bar plot.
            plot_type (str): One of ("normalized", "raw", "mixed") will use normalized data as default. "mixed" will
                plot one column of relative measure and one for the raw data. Note that mulitple columns are not
                supported when plotting mixed data.

        Returns: go.Figure

            A plotly figure containing a plot of the protein distribution  for each protein identified via the
            rapdorids.

            """
    if rapdordata.state.kernel_size is not None:
        i = int(rapdordata.state.kernel_size // 2)
    else:
        i = 0
    proteins = rapdordata[rapdorids]

    annotation = list(rapdordata.df[title_col][proteins])
    if "horizontal_spacing" not in kwargs:
        kwargs["horizontal_spacing"] = 0.15
    if "shared_xaxes" not in kwargs:
        if plot_type == "zoomed":
            kwargs["shared_xaxes"] = False
        else:
            kwargs["shared_xaxes"] = True
    if "column_titles" not in kwargs:
        if plot_type == "zoomed":
            kwargs["column_titles"] = [None, "Zoom to strongest shift"]
    if "rows" in kwargs and "cols" in kwargs:
        rows = kwargs["rows"]
        cols = kwargs["cols"]
        del kwargs["rows"]
        del kwargs["cols"]
    else:
        if plot_type == "zoomed":
            cols = 2
        elif plot_type == "mixed":
            cols = 2
        else:
            cols = 1
        rows = len(proteins)
    if "barmode" in kwargs:
        barmode = kwargs["barmode"]
        del kwargs["barmode"]
    else:
        barmode = "overlay"
    x1s = None
    if rows * cols < len(proteins):
        raise ValueError(f"Not enough columns ({cols}) and rows ({rows}) to place {len(proteins)} figures")
    if plot_type == "normalized":
        y_mode = "rel."
        y_title = f"{y_mode} {rapdordata.measure_type} {rapdordata.measure}"
        arrays = [rapdordata.norm_array[protein] for protein in proteins]
    elif plot_type == "raw":
        y_mode = "raw"
        y_title = f"{y_mode} {rapdordata.measure_type} {rapdordata.measure}"
        arrays = [rapdordata.kernel_array[protein] for protein in proteins]

    elif plot_type == "mixed":
        if cols != 2:
            raise ValueError("Number of columns not supported for mixed plot")
        y_title = None
        arrays = [rapdordata.norm_array[protein] for protein in proteins] + [rapdordata.kernel_array[protein] for
                                                                             protein in proteins]
        annotation += list(rapdordata.df[title_col][proteins])
    elif plot_type == "zoomed":
        arrays = [rapdordata.norm_array[protein] for protein in proteins] * 3
        y_title = f"rel. {rapdordata.measure_type} {rapdordata.measure}"
        annotation += list(rapdordata.df[title_col][proteins])
        x1s = [rapdordata.df.loc[protein]["position strongest shift"] for protein in proteins]
        if rapdordata.categorical_fraction:
            raise ValueError("This plot type is not supported for categorical fractions")
    else:
        raise ValueError("Plot type not supported")

    fig_subplots = make_subplots(
        rows=rows, cols=cols,
        x_title="Fraction",
        y_title=y_title,
        # row_titles=list(annotation),
        **kwargs
    )
    idx = 0
    clipmaxes = []
    for col_idx in range(cols):
        for row_idx in range(rows):
            if idx < len(arrays):
                array = arrays[idx]
                if x1s and col_idx == 0:
                    ma = x1s[idx] - i + zoom_fractions
                    mi = x1s[idx] - 1 - i - zoom_fractions
                    ma = int(min(ma, array.shape[-1]))
                    mi = int(max(mi, 0))
                    wmax = np.nanmax(array[:, mi:ma])
                    wmin = np.nanmin(array[:, mi:ma])
                    margin = (wmax - wmin) * 0.025
                    wmax = wmax + margin
                    wmin = wmin - margin
                    clipmaxes.append((wmin, wmax))

                plot_idx = (row_idx) * cols + (col_idx + 1) if idx != 0 else ""
                xref = f"x{plot_idx} domain"
                yref = f"y{plot_idx} domain"
                if mode == "line":
                    fig = plot_distribution(array, rapdordata.internal_design_matrix, offset=i, colors=colors)
                elif mode == "bar":
                    fig = plot_bars(array, rapdordata.internal_design_matrix, offset=i, colors=colors, barmode=barmode,
                                    x=rapdordata.fractions)

                else:
                    raise ValueError("mode must be one of line or bar")
                if plot_type not in ("mixed", "zoomed") or col_idx == 1:
                    fig_subplots.add_annotation(
                        text=annotation[idx],
                        xref=xref,
                        yref=yref,
                        x=1,
                        y=0.5,
                        yanchor="middle",
                        xanchor="left",
                        showarrow=False,
                        textangle=90
                    )
                for trace in fig["data"]:
                    if idx > 0:
                        trace['showlegend'] = False
                    fig_subplots.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
                idx += 1
    if mode == "bar":
        fig_subplots.update_layout(
            barmode="overlay",
            bargroupgap=0,
            bargap=0,
        )
        fig_subplots.update_xaxes(
            tickvals=list(range(len(rapdordata.fractions))),
            ticktext=[val.replace(" ", "<br>").replace("<br>&<br>", " &<br>") for val in rapdordata.fractions],
            tickmode="array",
            col=1
        )
    if plot_type == "mixed":
        fig_subplots.add_annotation(
            text=f"rel. {rapdordata.measure_type} {rapdordata.measure}",
            xref="paper",
            yref="paper",
            x=0,
            y=0.5,
            yanchor="middle",
            xanchor="right",
            showarrow=False,
            textangle=90,
            xshift=-40
        )
        fig_subplots.add_annotation(
            text=f"raw {rapdordata.measure_type} {rapdordata.measure}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            yanchor="middle",
            xanchor="left",
            showarrow=False,
            textangle=90,
            xshift=-20
        )

    if plot_type == "zoomed":
        for idx, entry in enumerate(x1s):
            clipmax = clipmaxes[idx]
            fig_subplots.update_xaxes(range=[entry - zoom_fractions, entry + zoom_fractions], row=idx + 1, col=2)
            fig_subplots.update_yaxes(row=idx + 1, col=2, range=clipmax)

    fig_subplots.update_layout(
        legend=fig["layout"]["legend"],
        legend2=fig["layout"]["legend2"],

    )
    return fig_subplots


def plot_var_histo(rapdorids, rapdordata: RAPDORData, color: str = DEFAULT_COLORS["primary"],
                   var_measure: str = "ANOSIM R", bins: int = 10):
    """
    Plots a histogram of a specified variability measure for a subset of proteins.

    Args:
        rapdorids (iterable):
            A list or set of RAPDOR IDs specifying the proteins to include in the histogram.
        rapdordata (RAPDORData):
            A RAPDORData object containing the full dataset, including a `.df` DataFrame
            with variability measures.
        color (str, optional):
            Hex or named color used to fill the histogram bars.
            Defaults to `DEFAULT_COLORS["primary"]`.
        var_measure (str, optional):
            Column name in `rapdordata.df` to use for the histogram values.
            Defaults to `"ANOSIM R"`.
        bins (int, optional):
            Number of bins to use in the histogram.
            Defaults to `10`.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly histogram figure showing the distribution of the selected variability measure.

    Example:
        fig = plot_var_histo(["P12345", "P67890"], rapdordata, var_measure="ANOSIM R", bins=20)
        fig.show()

    """
    fig = go.Figure()
    proteins = rapdordata[rapdorids]
    x = rapdordata.df.loc[proteins][var_measure]
    x_min = rapdordata.df[var_measure].min()
    x_max = rapdordata.df[var_measure].max()
    fig.add_trace(
        go.Histogram(
            x=x,
            marker_color=color,
            histnorm='probability',

            xbins=dict(
                start=x_min,
                end=x_max,
                size=(x_max - x_min) / bins,
            ),
        )
    )
    fig.update_xaxes(title=var_measure, range=[x_min, x_max])
    fig.update_yaxes(title="Freq.")
    return fig


def plot_distance_histo(rapdorids, rapdordata: RAPDORData, color: str = DEFAULT_COLORS["secondary"], bins: int = 10):
    """
    Plots a histogram of the mean pairwise distances for a subset of proteins.

    Args:
        rapdorids (iterable):
            A list or set of RAPDOR IDs specifying the proteins to include in the histogram.
        rapdordata (RAPDORData):
            A RAPDORData object containing the full dataset. Must include:
            - `df`: A DataFrame with a `"Mean Distance"` column.
            - `state.distance_method`: A string describing the distance metric used.
        color (str, optional):
            Hex or named color to use for the histogram bars.
            Defaults to `DEFAULT_COLORS["secondary"]`.
        bins (int, optional):
            Number of bins to divide the data into. Defaults to `10`.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly histogram figure showing the distribution of mean distances
            for the selected proteins.

    Example:
        fig = plot_distance_histo(["P12345", "P67890"], rapdordata, bins=20)
        fig.show()

    """
    fig = go.Figure()
    proteins = rapdordata[rapdorids]
    x = rapdordata.df.loc[proteins]["Mean Distance"]
    x_min = rapdordata.df["Mean Distance"].min() - 1e-10
    x_max = rapdordata.df["Mean Distance"].max() + 1e-10
    fig.add_trace(
        go.Histogram(
            x=x,
            marker_color=color,
            histnorm='probability',
            xbins=dict(
                start=x_min,
                end=x_max,
                size=(x_max - x_min) / bins,
            ),
        )
    )
    fig.update_xaxes(title=rapdordata.state.distance_method, range=[x_min, x_max])
    fig.update_yaxes(title="Freq.")
    return fig


def plot_mean_distributions(rapdorids, rapdordata: RAPDORData, colors, title_col: str = None):
    """Plots means of  distributions of the ids specified. Plots a Violin plot if fraction is categorical


    Args:
        rapdorids (List[any]): RAPDORids that should be plotted
        rapdordata (RAPDORData): a RAPDORData object containing the IDs from rapdorids
        colors (Iterable[str]): An iterable of color strings to use for plotting
        title_col (str): Name of a column that is present of the dataframe in rapdordata. Will use this column
            as names for the mean distributions. Set to None if Names should not be displayed.

    Returns: go.Figure()

    """
    if rapdordata.state.kernel_size is not None:
        i = int(rapdordata.state.kernel_size // 2)
    else:
        i = 0
    fig = go.Figure()
    proteins = rapdordata[rapdorids]
    if title_col is not None:
        annotation = list(rapdordata.df[title_col][proteins])
    else:
        annotation = ["" for _ in range(len(proteins))]

    indices = rapdordata.indices
    levels = rapdordata.treatment_levels
    x = rapdordata.fractions
    x = x[i: rapdordata.norm_array.shape[-1] + i]

    names = []
    j_val = max([len(name) for name in levels])

    for eidx, (orig_name, idx) in enumerate(zip(levels, indices)):
        name = f"{orig_name}".ljust(j_val, " ")
        legend = f"legend{eidx + 1}"
        names.append(name)
        overall_means = []
        for pidx, protein in enumerate(proteins):
            subdata = rapdordata.norm_array[protein]
            mean_values = np.nanmean(subdata[idx,], axis=0)
            showlegend = False if title_col is None else True
            color = _color_to_calpha(colors[eidx], 0.25)
            fig.add_trace(go.Scatter(
                x=x,
                y=mean_values,
                marker=dict(color=color),
                name=annotation[pidx],
                mode="lines",
                legend=legend,
                line=dict(width=1),
                showlegend=showlegend,
                legendgroup=legend if title_col is None else None

            ))
            overall_means.append(mean_values)
        overall_means = np.asarray(overall_means)
        if not rapdordata.categorical_fraction:
            overall_means = overall_means.mean(axis=0)

            fig.add_trace(go.Scatter(
                x=x,
                y=overall_means,
                marker=dict(color=colors[eidx]),
                mode="lines",
                name=f"{orig_name} mean" if title_col is not None else "",
                legend=legend,
                line=dict(width=2),
                showlegend=True,
                legendgroup=legend if title_col is None else None

            ))
        else:
            x_box = np.asarray([x for _ in range(overall_means.shape[0])])
            fig.add_trace(go.Violin(
                y=overall_means.flatten(),
                x=x_box.flatten(),
                name=f"{orig_name} box" if title_col is not None else "",
                marker=dict(color=colors[eidx]),
                legend=legend,
                line=dict(width=2),
                showlegend=True,
                legendgroup=legend if title_col is None else None,
                side="positive" if eidx == 0 else "negative",
                box_visible=True,
                meanline_visible=True,
                spanmode="hard",
                width=0.5
            ))
    fig.update_layout(violingap=0, violinmode='overlay')

    fig = _update_distribution_layout(fig, names, x, i, yname=f"rel. {rapdordata.measure_type} {rapdordata.measure}")
    return fig


def plot_means_and_histos(rapdorids, rapdordata: RAPDORData, colors, title_col: str = None, **kwargs):
    """
    Combines mean distribution plots and histograms into a single figure with subplots.

    This function generates a vertically stacked composite figure containing:
    1. Mean expression distributions,
    2. A histogram of mean distances,
    3. A histogram of a variability measure (e.g., ANOSIM R).

    Args:
        rapdorids (iterable):
            A list or set of RAPDOR IDs for which data will be plotted.
        rapdordata (RAPDORData):
            A RAPDORData object that contains:
            - `df`: A pandas DataFrame with PCA and metric data.
            - `state.distance_method`: A string for labeling the distance histogram.
        colors (list or tuple):
            A sequence of color values. The first is used for the distance histogram,
            the second for the variability histogram, and passed into mean plotting.
        title_col (str, optional):
            Name of the column in `rapdordata.df` used for labeling mean distribution plots.
        **kwargs:
            Additional keyword arguments passed to `make_subplots`, such as:
            - `row_heights` (list of float): Heights for the three subplot rows.
              Defaults to `[0.5, 0.25, 0.25]`.
            - `vertical_spacing` (float): Spacing between subplot rows. Defaults to `0.1`.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly figure object with three vertically stacked subplots.

    Example:
        fig = plot_means_and_histos(["P12345", "P67890"], rapdordata, colors=["#1f77b4", "#ff7f0e"])
        fig.show()

    """
    if "row_heights" not in kwargs:
        kwargs["row_heights"] = [0.5, 0.25, 0.25]
    if "vertical_spacing" not in kwargs:
        kwargs["vertical_spacing"] = 0.1

    fig = make_subplots(rows=3, cols=1, **kwargs)

    fig1 = plot_mean_distributions(rapdorids, rapdordata, colors, title_col)
    fig2 = plot_distance_histo(rapdorids, rapdordata, colors[0])
    fig3 = plot_var_histo(rapdorids, rapdordata, colors[1])
    for trace in fig1['data']:
        fig.add_trace(trace, row=1, col=1)
        fig.update_xaxes(fig1["layout"]["xaxis"], row=1)
        fig.update_yaxes(fig1["layout"]["yaxis"], row=1)

    for trace in fig2['data']:
        trace.update(showlegend=False)
        fig.add_trace(trace, row=2, col=1)
        fig.update_xaxes(fig2["layout"]["xaxis"], row=2)
        fig.update_yaxes(fig2["layout"]["yaxis"], row=2)

    # Add traces from the second figure to the second subplot
    for trace in fig3['data']:
        trace.update(showlegend=False)
        fig.add_trace(trace, row=3, col=1)
        fig.update_xaxes(fig3["layout"]["xaxis"], row=3)
        fig.update_yaxes(fig3["layout"]["yaxis"], row=3)

    # Add traces from the third figure to the third subplot

    fig.update_layout(
        legend=fig1["layout"]["legend"],
        legend2=fig1["layout"]["legend2"],

    )

    return fig


def _coordinates_to_svg_path(x_coords, y_coords):
    if len(x_coords) != 3 or len(y_coords) != 3:
        raise ValueError("The input should contain exactly 3 coordinates for a triangle.")

    path = f"M {x_coords[0]} {y_coords[0]} "  # Move to the first vertex

    for i in range(1, 3):
        path += f"L {x_coords[i]} {y_coords[i]} "  # Draw lines to the other vertices

    path += "Z"  # Close the path

    return path


def rank_plot(
    rapdorsets: Dict[str, Iterable],
    rapdordata: RAPDORData,
    colors,
    orientation: str = "h",
    triangles: str = "inside",
    tri_x: float = 25,
    tri_y: float = 0.1,
    label_col: str = "RAPDORid"
):
    """
    Creates a bar-based rank plot to visualize the positions of RAPDOR sets within a ranked list.

    Each RAPDOR set is displayed as a bar indicating presence at a given rank, with an optional
    triangle annotation marking the median rank of the set. The plot can be rendered in either
    horizontal or vertical orientation.

    Args:
        rapdorsets (Dict[str, Iterable]):
            A dictionary mapping set names to iterables of RAPDOR IDs to be plotted.
        rapdordata (RAPDORData):
            A RAPDORData object containing a `.df` DataFrame with at least:
            - `"Rank"`: Numerical rank values for each entry.
            - `"RAPDORid"`: Identifiers matching those in `rapdorsets`.
            - `label_col`: A column used for hover tooltips (e.g., gene names).
        colors (list or tuple):
            A list of color values (hex strings or named colors), one per RAPDOR set.
        orientation (str, optional):
            Orientation of the plot. `"h"` for horizontal (default), `"v"` for vertical.
        triangles (str, optional):
            Whether to draw triangle indicators for median rank positions.
            `"inside"` places the triangles inside the plot area;
            `"outside"` places them in the plot margin.
            Defaults to `"inside"`.
        tri_x (float, optional):
            Half-width of the triangle in the X direction (or Y if orientation is vertical).
            Defaults to `25`.
        tri_y (float, optional):
            Height of the triangle. For `"outside"`, this is a vertical offset.
            Defaults to `0.1`.
        label_col (str, optional):
            Name of the column in `rapdordata.df` to use for hover text (e.g., "Gene", "ProteinName").
            Defaults to `"Gene"`.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly figure displaying the rank positions of the specified RAPDOR sets.

    Example:
        fig = rank_plot(
            rapdorsets={"Set A": ["P123", "P456"]},
            rapdordata=rapdordata,
            colors=["#1f77b4"],
            label_col="ProteinName"
        )
        fig.show()

    """
    fig = go.Figure(layout=dict(template=DEFAULT_TEMPLATE))
    df = rapdordata.df.sort_values(by="Rank")
    df = df[~pd.isna(df["Rank"])]
    init_x = list(range(int(df["Rank"].min()), int(df.Rank.max()) + 1))
    if triangles == "outside":
        trirefx = "paper" if orientation == "v" else "x"
        trirefy = "paper" if orientation == "h" else "y"
        anoxanchor = "center" if orientation == "h" else "right"
        anoyanchor = "top" if orientation == "h" else "middle"

        modifier = tri_y
    else:
        trirefx, trirefy = "x", "y"
        modifier = 0
        anoxanchor = "center" if orientation == "h" else "left"
        anoyanchor = "bottom" if orientation == "h" else "middle"

    for idx, (key, data) in enumerate(rapdorsets.items()):
        df = df.sort_values(by="Rank")
        data = df[df["RAPDORid"].isin(data)]["Rank"] - 1
        x = init_x
        y = np.empty(len(x))
        y.fill(np.nan)
        y[data] = 1.
        tri = np.nanmedian((y * np.asarray(x)))
        trixs = [tri - tri_x, tri, tri + tri_x]
        triys = [0 - modifier, tri_y - modifier, 0 - modifier]
        if orientation == "v":
            triys, trixs = trixs, triys
            x, y = y, x
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                marker_color=colors[idx],
                marker_line=dict(width=2, color=colors[idx]),
                name=key,
                hovertext=df[label_col],
                orientation="v" if orientation == "h" else "h"

            )
        )
        color = fig.layout.xaxis.linecolor
        fig.add_shape(
            dict(
                type="path",
                path=_coordinates_to_svg_path(trixs, triys),
                line_color=color if color else "black",
                fillcolor=colors[idx],
                xref=trirefx,
                yref=trirefy,
                name=f"{key} median",
                layer="below" if triangles == "outside" else None,
                showlegend=True

            )
        )
        fig.add_annotation(
            text=f"{tri:.1f}",
            xref=trirefx,
            yref=trirefy,
            x=trixs[1] if orientation == "h" else trixs[0],
            y=triys[0] if orientation == "h" else triys[1],
            showarrow=False,
            xanchor=anoxanchor,
            yanchor=anoyanchor,

        )

    fig.update_layout(barmode="overlay", bargap=0,
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=1, xanchor="right"))
    if orientation == "h":
        fig.update_xaxes(title="Rank", range=(init_x[0], init_x[-1]), showgrid=True)
        fig.update_yaxes(range=(0, 1), showgrid=False, showticklabels=False)
    elif orientation == "v":
        fig.update_yaxes(title="Rank", range=(init_x[0], init_x[-1]), showgrid=True)
        fig.update_xaxes(range=(0, 1), showgrid=False, showticklabels=False)

    fig.layout.xaxis.type = "linear"
    fig.layout.yaxis.type = "linear"
    return fig


def multi_means_and_histo(rapdorsets: Dict[str, Iterable], rapdordata: RAPDORData, colors, **kwargs):
    """Plots histograms of ANOSIM R and Distance as well as the distributions of the mean of multiple ids.

    Args:
        rapdorsets (dict): a dictionary containing a key that will appear in the plot as column header.
            The values of the dictionary must be a list that contains ids from the RAPDORData used in rapdordata.
        rapdordata (RAPDORData): a RAPDORData object containing the IDs from rapdorids
        colors (Iterable[str]): An iterable of color strings to use for plotting. Muste have length 3. Will use the
            first two colors for the distribution and the third for the histograms.
        **kwargs: Will be passed to the make_subplots call of plotly

    Returns: go.Figure

    """
    if "row_heights" not in kwargs:
        kwargs["row_heights"] = [0.15, 0.15, 0.7]
    if "vertical_spacing" not in kwargs:
        kwargs["vertical_spacing"] = 0.06
    if "horizontal_spacing" not in kwargs:
        kwargs["horizontal_spacing"] = 0.02
    if "row_titles" not in kwargs:
        distance = "JSD" if rapdordata.state.distance_method == "Jensen-Shannon-Distance" else "Distance"
        kwargs["row_titles"] = [distance, "ANOSIM R", "Distribution"]
    if "column_titles" not in kwargs:
        kwargs["column_titles"] = list(rapdorsets.keys())
    if "x_title" not in kwargs:
        kwargs["x_title"] = "Fraction"

    fig = make_subplots(rows=3, cols=len(rapdorsets), shared_yaxes=True, **kwargs)

    for c_idx, (name, rapdorids) in enumerate(rapdorsets.items(), 1):
        fig1 = plot_distance_histo(rapdorids, rapdordata, colors[2])
        fig2 = plot_var_histo(rapdorids, rapdordata, colors[2])
        fig3 = plot_mean_distributions(rapdorids, rapdordata, colors)

        for trace in fig1['data']:
            trace.update(showlegend=False)
            fig.add_trace(trace, row=1, col=c_idx)
            fig.update_xaxes(fig1["layout"]["xaxis"], row=1, col=c_idx)
            fig.update_yaxes(fig1["layout"]["yaxis"], row=1, col=c_idx)

        # Add traces from the second figure to the second subplot
        for trace in fig2['data']:
            trace.update(showlegend=False)
            fig.add_trace(trace, row=2, col=c_idx)
            fig.update_xaxes(fig2["layout"]["xaxis"], row=2, col=c_idx)
            fig.update_yaxes(fig2["layout"]["yaxis"], row=2, col=c_idx)

        for trace in fig3['data']:
            if c_idx > 1:
                trace.update(showlegend=False)
            fig.add_trace(trace, row=3, col=c_idx)
            fig.update_xaxes(fig3["layout"]["xaxis"], row=3, col=c_idx)
            fig.update_xaxes(title=None, row=3, col=c_idx)
            fig.update_yaxes(fig3["layout"]["yaxis"], row=3)

    fig.update_layout(violingap=0, violinmode='overlay', bargap=0)
    fig.update_xaxes(title=None, row=1)
    fig.update_xaxes(title=None, row=2)
    fig.update_yaxes(title=None, col=2)
    fig.update_yaxes(title=None, col=3)

    # Add traces from the third figure to the third subplot

    fig.update_layout(
        legend=fig3["layout"]["legend"],
        legend2=fig3["layout"]["legend2"],

    )
    fig.update_layout(
        legend2=dict(y=-0.05, yref="paper", yanchor="top", font=None),
        legend=dict(y=-0.1, yref="paper", yanchor="top", font=None),

    )

    fig.update_layout(
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        legend2=dict(bgcolor='rgba(0,0,0,0)'),

    )
    return fig


def plot_bars(subdata, design, x, offset: int = 0, colors=None, yname: str = "rel. protein amount", barmode: str = "overlay"):
    """
    Plots bar charts with overlaid scatter markers and quantile-based error bars
    for grouped protein abundance data across experimental conditions.

    This function groups samples by treatment (as specified in the `design` DataFrame),
    computes mean and interquartile range (25–75%) for each group, and visualizes the results
    using Plotly bars and error bars.

    Args:
        subdata (np.ndarray):
            A 2D NumPy array with shape (n_samples, n_features), containing protein abundance
            values. Sample indices must align with the `design` DataFrame.
        design (pd.DataFrame):
            A DataFrame that includes a `"Treatment"` column mapping sample indices to group labels.
        x (list):
            A list of original x-axis labels corresponding to features (e.g., protein IDs or timepoints).
        offset (int, optional):
            Index offset applied to the `x` list to align with `subdata` columns. Default is `0`.
        colors (list, optional):
            A list of color values to be used for each treatment group. If `None`, default colors are used.
        yname (str, optional):
            Y-axis label for the plot. Defaults to `"rel. protein amount"`.
        barmode (str, optional):
            Bar layout mode for Plotly. Options include `"overlay"` and `"group"`. Default is `"overlay"`.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly figure containing the bar chart with overlaid quantile scatter markers and error bars.

    Notes:
        - For each group in `design["Treatment"]`, the function computes:
            - Mean protein abundance across group samples.
            - 25th and 75th quantiles to derive error bars.
        - Each group is visualized using a combination of bars and markers.
        - Hover mode is set to `"x"` for better interactivity.

    Example:
        fig = plot_bars(
            subdata=protein_array,
            design=sample_design_df,
            x=["P1", "P2", "P3"],
            colors=["#1f77b4", "#ff7f0e"]
        )
        fig.show()

    """
    if colors is None:
        colors = DEFAULT_COLORS
    fig = go.Figure(layout=go.Layout(yaxis2=go.layout.YAxis(
        visible=False,
        matches="y",
        overlaying="y",
        anchor="x",
    )))
    indices = design.groupby("Treatment", group_keys=True).apply(lambda x: list(x.index))
    origininal_x = x[offset: subdata.shape[1] + offset]
    x = list(range(len(origininal_x)))
    print(x)

    names = []
    j_val = max([len(name) for name in indices.keys()])

    for eidx, (name, idx) in enumerate(indices.items()):
        name = f"{name}".ljust(j_val, " ")
        legend = f"legend{eidx + 1}"
        offset = -0.05 if eidx else 0.05
        offset_x = [v+offset for v in x]
        names.append(name)
        mean_values = np.nanmean(subdata[idx,], axis=0)
        upper_quantile = np.nanquantile(subdata[idx,], 0.75, axis=0) - mean_values
        lower_quantile = mean_values - np.nanquantile(subdata[idx,], 0.25, axis=0)
        fig.add_trace(go.Bar(
            y=mean_values,
            x=offset_x if barmode != "overlay" else x,
            name="Bar",
            offsetgroup=str(eidx),
            # offset=(eidx - 1) * 1 / 3,
            legend=legend,
            marker=dict(color=colors[eidx], opacity=0.5,),
            width=.1 if barmode != "overlay" else None
        ))
        fig.add_trace(
            go.Scatter(
                x=offset_x,
                y=mean_values,
                offsetgroup=str(eidx),
                mode="markers",
                marker=dict(color=colors[eidx], size=20),
                error_y = dict(
                    type='data',  # value of error bar given in data coordinates
                    array=upper_quantile,
                    arrayminus=lower_quantile,
                    symmetric=False,
                    visible=True,
                    color=colors[eidx],
                    thickness=2,
                    width=10
                ),
                legend=legend,
                name="Dot & IQR"

            )
        )
    fig.update_layout(hovermode="x")
    fig.update_layout(
        yaxis_title=yname,
    )
    fig.update_layout(
        bargroupgap=0,
        bargap=0,
        barmode="overlay"
    )
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=x,
            ticktext=origininal_x
        )
    )
    fig = _update_distribution_layout(fig, names, x, offset, yname)
    return fig


def plot_distribution(
        subdata,
        design: pd.DataFrame,
        offset: int = 0,
        colors: Iterable = None,
        yname: str = "rel. protein amount",
        show_outliers: bool = True
):
    """Plots the distribution of proteins using mean, median, min and max values of replicates

        Args:
            subdata (np.ndarray): an array of shape :code:`num samples x num_fractions`.
            design (pd.Dataframe): the design dataframe to distinguish the groups from the samples dimension
            offset (int): adds this offset to the fractions at the x-axis range
            colors (Iterable[str]): An iterable of color strings to use for plotting
            yname (str): yaxis_title
            show_outliers (bool): Shows min and max of subdata.

        Returns: go.Figure

            A plotly figure containing a scatter-line for the mean, median, min and max of
            the replicates.

        """
    if colors is None:
        colors = DEFAULT_COLORS
    fig = go.Figure()
    indices = design.groupby("Treatment", group_keys=True, observed=False).apply(lambda x: list(x.index))
    medians = []
    means = []
    errors = []
    if offset:
        x = list(range(offset + 1, subdata.shape[1] + offset + 1))
    else:
        x = list(range(subdata.shape[1]))
    names = []
    j_val = max([len(name) for name in indices.keys()])
    for eidx, (name, idx) in enumerate(indices.items()):
        name = f"{name}".ljust(j_val, " ")
        legend = f"legend{eidx + 1}"
        names.append(name)
        median_values = np.nanmedian(subdata[idx,], axis=0)

        mean_values = np.nanmean(subdata[idx,], axis=0)
        upper_quantile = np.nanquantile(subdata[idx,], 0.75, axis=0)
        lower_quantile = np.nanquantile(subdata[idx,], 0.25, axis=0)
        color = colors[eidx]
        a_color = _color_to_calpha(color, 0.4)
        a_color2 = _color_to_calpha(color, 0.15)
        medians.append(go.Scatter(
            x=x,
            y=median_values,
            marker=dict(color=colors[eidx]),
            name="Median",
            legend=legend,
            line=dict(dash="dot"),
            mode="lines"

        ))
        means.append(go.Scatter(
            x=x,
            y=mean_values,
            marker=dict(color=colors[eidx]),
            name="Mean",
            legend=legend,
            line=dict(width=3)

        ))
        y = np.concatenate((upper_quantile, np.flip(lower_quantile)), axis=0)
        if show_outliers:
            max_values = np.nanmax(subdata[idx,], axis=0)
            min_values = np.nanmin(subdata[idx,], axis=0)
            outliers = np.concatenate((max_values, np.flip(min_values)), axis=0)
            errors.append(
                go.Scatter(
                    x=x + x[::-1],
                    y=outliers,
                    marker=dict(color=colors[eidx]),
                    name="Min-Max",
                    legend=legend,
                    fill="tonexty",
                    fillcolor=a_color2,
                    line=dict(color='rgba(255,255,255,0)')
                )
            )
        errors.append(
            go.Scatter(
                x=x + x[::-1],
                y=y,
                marker=dict(color=colors[eidx]),
                name="IQR",
                legend=legend,
                fill="toself",
                fillcolor=a_color,
                line=dict(color='rgba(255,255,255,0)')
            )
        )
    fig.add_traces(
        errors
    )
    fig.add_traces(
        medians + means
    )
    fig = _update_distribution_layout(fig, names, x, offset, yname)
    return fig


def _update_distribution_layout(fig, names, x, offset, yname):
    fig.update_layout(hovermode="x")
    if not isinstance(x[0], str):
        fig.update_layout(xaxis_range=[x[0] - offset - 0.5, x[-1] + offset + 0.5])
    fig.update_layout(
        yaxis_title=yname,
    )
    fig.update_layout(
        xaxis=dict(title="Fraction"),
        legend=dict(
            title=names[0],
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            itemsizing="constant",
            font=dict(family='SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace', size=16)

        ),
        legend2=dict(
            title=names[1],
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0,
            itemsizing="constant",
            font=dict(family='SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace', size=16)

        )
    )
    fig.update_layout(
        showlegend=True
    )
    return fig


def plot_heatmap(distances, design: pd.DataFrame, colors=None):
    """Plots a heatmap of the sample distances

    Args:
        distances (np.ndarray): between sample distances of shape :code:`num samples x num samples`
        design (pd.Dataframe): the design dataframe to distinguish the groups from the samples dimension
        colors (Iterable[str]): An iterable of color strings to use for plotting

    Returns: go.Figure

    """
    if colors is None:
        colors = DEFAULT_COLORS
    names = design["Treatment"].astype(str) + " " + design["Replicate"].astype(str)
    fig = go.Figure(
        data=go.Heatmap(
            z=distances,
            x=names,
            y=names,
            colorscale=colors[:2]
        )
    )
    fig.update_yaxes(showgrid=False, mirror=True, showline=True, linecolor="black", linewidth=2)
    fig.update_xaxes(showgrid=False, mirror=True, showline=True, linecolor="black", linewidth=2)
    return fig


def plot_protein_westernblots(rapdorids, rapdordata: RAPDORData, colors, title_col: str = "RAPDORid",
                              vspace: float = 0.01, scale_max: bool = True):
    """Plots a figure containing a pseudo westernblot of the protein distribution.

    This will ignore smoothing kernels and plots raw mean replicate intensities.
    It will also normalize subplot colors based on the maximum intensity.

    Args:
        rapdorids (List[any]): RAPDORids that should be plotted
        rapdordata (RAPDORData): a RAPDORData object containing the IDs from rapdorids
        colors (Iterable[str]): An iterable of color strings to use for plotting
        title_col (str): Name of a column that is present of the dataframe in rapdordata. Will add this column
            as a subtitle in the plot (Default: RAPDORid)
        vspace (float): Vertical space between subplots

    Returns: go.Figure

        A plotly figure containing a pseudo westernblot of the protein distribution  for each protein identified via the
        rapdorids.

    """

    proteins = rapdordata[rapdorids]
    annotation = rapdordata.df[title_col][proteins].repeat(2)
    fig_subplots = make_subplots(rows=len(proteins) * 2, cols=1, shared_xaxes=True, x_title="Fraction",
                                 row_titles=list(annotation), vertical_spacing=0.0,
                                 specs=[
                                     [
                                         {
                                             "t": vspace / 2 if not idx % 2 else 0.000,
                                             "b": vspace / 2 if idx % 2 else 0.000
                                         }
                                     ] for idx in range(len(proteins) * 2)
                                 ]

                                 )
    for idx, protein in enumerate(proteins, 1):
        array = rapdordata.array[protein]
        fig = plot_barcode_plot(array, rapdordata.internal_design_matrix, colors=colors, fractions=rapdordata.fractions,
                                scale_max=scale_max)
        for i_idx, trace in enumerate(fig["data"]):
            fig_subplots.add_trace(trace, row=(idx * 2) + i_idx - 1, col=1)
    fig = fig_subplots
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        marker=dict(color=colors[0], symbol="square"),
        showlegend=True,
        mode="markers",
        name=fig.data[0].name

    )),
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        marker=dict(color=colors[1], symbol="square"),
        showlegend=True,
        mode="markers",
        name=fig.data[1].name,
    ))
    fig.update_layout(
        legend=dict(
            itemsizing="constant",
            x=0,
            y=1,
            yanchor="bottom"
        )
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_yaxes(showgrid=False, showline=True, linewidth=2, mirror=True)
    fig.update_xaxes(showticklabels=True, row=len(proteins) * 2, col=1)
    v = 1 / len(proteins)
    for idx in range(len(proteins) * 2):
        fig.update_xaxes(showgrid=False, showline=True, linewidth=2, side="top" if not idx % 2 else "bottom",
                         row=idx + 1, col=1)
        if scale_max:
            show = True if idx % 2 else False
            ticklabelpos = "outside"
        else:
            show = True
            ticklabelpos = "inside"

        fig.data[idx].colorbar.update(
            len=v,
            yref="paper",
            y=1 - v * (idx // 2),
            yanchor="top",
            nticks=3,
            x=1. + 0.05 if idx % 2 else 1.,
            showticklabels=show,
            thickness=0.05,
            thicknessmode="fraction",
            ticklabelposition=ticklabelpos,
            tickfont=dict(color=None)

        )
        y_domain = f"y{idx + 1} domain" if idx != 0 else "y domain"
        if not idx % 2:
            fig["layout"]["annotations"][idx].update(y=0, yref=y_domain, x=-0.05, textangle=270)
        else:
            fig["layout"]["annotations"][idx].update(text="")

    return fig_subplots


def plot_barcode_plot(subdata, design: pd.DataFrame, colors=None, vspace: float = 0.025, fractions=None,
                      scale_max: bool = True):
    """Creates a Westernblot like plot from the mean of protein intensities

    Args:
        subdata (np.ndarray): an array of shape :code:`num samples x num_fractions`. Rows don´t need to add up to one
        design (pd.Dataframe): the design dataframe to distinguish the groups from the samples dimension
        offset (int): adds this offset to the fractions at the x-axis range
        colors (Iterable[str]): An iterable of color strings to use for plotting
        vspace (float): vertical space between westernblots (between 0 and 1)

    Returns: go.Figure

        A figure containing two subplots of heatmaps of the non normalized intensities.

    """
    if colors is None:
        colors = DEFAULT_COLORS
    indices = design.groupby("Treatment", group_keys=True, observed=False).apply(lambda x: list(x.index))
    fig = make_subplots(cols=1, rows=2, vertical_spacing=vspace)

    ys = []
    scale = []
    means = []
    xs = []
    names = []
    for eidx, (name, idx) in enumerate(indices.items()):
        color = colors[eidx]
        a_color = _color_to_calpha(color, 0.)
        color = _color_to_calpha(color, 1)
        scale.append([[0, a_color], [1, color]])
        name = f"{name}"
        mean_values = np.mean(subdata[idx,], axis=0)
        ys.append([name for _ in range(len(mean_values))])
        if fractions is None:
            xs.append(list(range(1, subdata.shape[1] + 1)))
        else:
            xs.append(fractions)
        names.append(name)
        means.append(mean_values)
    m_val = [np.max(a) for a in means]

    if scale_max:
        m_val = max(m_val)
        m_val = [m_val, m_val]
    for idx, (x, y, z) in enumerate(zip(xs, ys, means)):
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale=scale[idx],
                name=names[idx],
                hovertemplate='<b>Fraction: %{x}</b><br><b>Protein Intensity: %{z:.2e}</b> ',

            ),
            row=idx + 1, col=1
        )
    fig.data[0].update(zmin=0, zmax=m_val[0])
    fig.data[1].update(zmin=0, zmax=m_val[1])
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, mirror=True, showline=True, linecolor="black", linewidth=2)
    fig.update_xaxes(showgrid=False, mirror=True, showline=True, linecolor="black", linewidth=2)
    fig.update_xaxes(title="Fraction", row=2, col=1)
    fig.data[0].colorbar.update(
        len=0.5,
        yref="paper",
        y=1,
        yanchor="top"

    )
    fig.data[1].colorbar.update(
        len=0.5,
        yref="paper",
        y=0.5,
        yanchor="top",

    )

    return fig


def plot_dimension_reduction(
        rapdordata: RAPDORData,
        colors: Iterable[str] = None,
        highlight: Iterable[Any] = None,
        highlight_color: str = None,
        show_cluster: bool = False,
        dimensions: int = 2,
        marker_max_size: int = 40,
        second_bg_color: str = "white",
        bubble_legend_color: str = "black",
        legend_start: float = 0.2,
        legend_spread: float = 0.1,
        title_col: str = None,
        cutoff_range: Tuple[float, float] = None,
        cutoff_type: str = None,
        show_lfc: bool = False,
):
    """Plots a bubble plot using relative distribution change, relative fraction shift and the Mean Distance

    Args:
        rapdordata (RAPDORData): A :class:`~RAPDOR.datastructures.RAPDORData` object where distances
            are calculated and the array is normalized already.
        colors (Iterable[str]): An iterable of color strings to use for plotting
        highlight (Iterable[Any]): RAPDORids to highlight in the plot
        highlight_color (str): html color used to highlight selected bubbles
        show_cluster (bool): If set to true it will show clusters in different colors.
            (Only works if rapdordata is clustered)
        dimensions (int): Either 2 or 3. 2 will produce a plot where the mean distance axis is represented via a marker
            size. If 3, it will add another axis and return a three-dimensional figure
        marker_max_size (int): maximum marker size for highest mean distance (This has no effect if dimensions is 3)
        second_bg_color (str): changes background color of the bubble legend (This has no effect if dimensions is 3)
        bubble_legend_color (str): color of the bubbles and text in the bubble legend
            (This has no effect if dimensions is 3)
        legend_start (float): start position of the bubble legend in relative coordinates
            (This has no effect if dimensions is 3)
        legend_spread (float): spread of the bubble legend (This has no effect if dimensions is 3)
        title_col (str): Will display names from that column in the rapdordata for highlighted proteins
            (This has no effect if dimensions is 3)
        cutoff_type (str): column in the dataframe of :class:`~RAPDOR.datastructures.RAPDORData` used for cutoff. Muste
            be numerical
        cutoff_range (Iterable[float]): uses index one as lower bound and index two as upper bound for data to display in
            the plot
        show_lfc (bool): Shows lfs via a colorscale and the bubble colors. No effect if dimension is 3


    Returns: go.Figure()

    """
    colors = COLORS + list(colors)
    if highlight is not None:
        highlight = rapdordata[highlight]
    if show_cluster:
        clusters = rapdordata.df["Cluster"]
    else:
        clusters = None

    if dimensions == 2:

        fig = _plot_dimension_reduction_result2d(
            rapdordata,
            colors,
            clusters,
            highlight,
            marker_max_size,
            second_bg_color,
            bubble_legend_color,
            legend_start,
            legend_spread,
            title_col,
            cutoff_type=cutoff_type,
            cutoff_range=cutoff_range,
            highlight_color=highlight_color,
            show_lfc=show_lfc,
        )
    elif dimensions == 3:
        fig = _plot_dimension_reduction_result3d(
            rapdordata,
            colors=colors,
            clusters=clusters,
            highlight=highlight,
            cutoff_type=cutoff_type,
            cutoff_range=cutoff_range
        )
    else:
        raise ValueError("Unsupported dimensionality")
    return fig


def _plot_dimension_reduction_result3d(rapdordata, colors=None, clusters=None, highlight=None, cutoff_range = None, cutoff_type = None):
    embedding = rapdordata.current_embedding

    fig = go.Figure()
    clusters = np.full(embedding.shape[0], -1) if clusters is None else clusters
    cutoff_mask = np.zeros(embedding.shape[0], dtype=bool)


    n_cluster = int(np.nanmax(clusters)) + 1
    mask = np.ones(embedding.shape[0], dtype=bool)
    hovertext = rapdordata.df.index.astype(str) + ": " + rapdordata.df["RAPDORid"].astype(str)
    data = rapdordata.df["Mean Distance"].to_numpy()
    if cutoff_range is not None:
        assert cutoff_type is not None
        indices = rapdordata.df[
            (rapdordata.df[cutoff_type] <= cutoff_range[1]) & (rapdordata.df[cutoff_type] >= cutoff_range[0])].index
        cutoff_mask[indices] = 1
    else:
        cutoff_mask = np.ones(embedding.shape[0], dtype=bool)

    if highlight is not None and len(highlight) > 0:
        indices = np.asarray([rapdordata.df.index.get_loc(idx) for idx in highlight])
        mask[indices] = 0
    if n_cluster > len(colors) - 2:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            text="Too Many Clusters<br> Will not show all<br>Please adjust cluster Settings",
            showarrow=False,
            font=(dict(size=28))
        )
    if np.any(clusters == -1):
        c_mask = mask & (clusters == -1) & cutoff_mask
        fig.add_trace(go.Scatter3d(
            x=embedding[c_mask, :][:, 0],
            y=embedding[c_mask, :][:, 1],
            z=data[c_mask],
            mode="markers",
            hovertext=hovertext[c_mask],
            marker=dict(color=colors[-2], size=4),
            name=f"Not Clustered",
        ))
        nmask = ~mask & (clusters == -1) & cutoff_mask
        fig.add_trace(
            go.Scatter3d(
                x=embedding[nmask, :][:, 0],
                y=embedding[nmask, :][:, 1],
                z=data[nmask],
                mode="markers",
                hovertext=hovertext[nmask],
                marker=dict(color=colors[-2], size=8, line=dict(color=colors[-1], width=4)),
                name="Not Clustered",

            )
        )
    for color_idx, cluster in enumerate(range(min(n_cluster, len(colors) - 2))):
        c_mask = mask & (clusters == cluster) & cutoff_mask
        fig.add_trace(go.Scatter3d(
            x=embedding[c_mask, :][:, 0],
            y=embedding[c_mask, :][:, 1],
            z=data[c_mask],
            mode="markers",
            hovertext=hovertext[c_mask],
            marker=dict(color=colors[color_idx], size=4),
            name=f"Cluster {cluster}"
        ))
        nmask = ~mask & (clusters == cluster) & cutoff_mask
        fig.add_trace(
            go.Scatter3d(
                x=embedding[nmask, :][:, 0],
                y=embedding[nmask, :][:, 1],
                z=data[nmask],
                mode="markers",
                hovertext=hovertext[nmask],
                marker=dict(color=colors[color_idx], size=8, line=dict(color=colors[-1], width=4)),
                name=f"Cluster {cluster}"
            )
        )

    fig.update_layout(
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title=f"relative fraction shift"),
            yaxis=go.layout.scene.YAxis(title=f"relative distribution change"),
            zaxis=go.layout.scene.ZAxis(title=f"Mean Distance"),
        )

    )
    return fig


def update_bubble_legend(fig, legend_start: float = 0.2, legend_spread: float = 0.1, second_bg_color: str = None,
                         bubble_legend_color: str = None):
    xloc = [legend_start + idx * legend_spread for idx in range(3)]
    fig.data[0].x = xloc
    annos = [annotation for annotation in fig.layout.annotations if
             (annotation.text != "Mean Distance" and annotation.xref == "x")]
    if second_bg_color is not None:
        fig.update_shapes(fillcolor=second_bg_color)
    for idx, annotation in enumerate(annos):
        annotation.update(
            x=xloc[idx] + 0.02 + idx * 0.02 / 3,
        )

    if bubble_legend_color is not None:
        fig.data[0].update(marker=dict(line=dict(color=bubble_legend_color)))
    return fig


def _plot_dimension_reduction_result2d(rapdordata: RAPDORData, colors=None, clusters=None,
                                       highlight=None, marker_max_size: int = 40, second_bg_color: str = "white",
                                       bubble_legend_color: str = "black", legend_start: float = 0.2,
                                       legend_spread: float = 0.1,
                                       sel_column=None, cutoff_range: Tuple[float, float] = None,
                                       cutoff_type: str = None, highlight_color: str = None, show_lfc: bool = False,

                                       ):
    embedding = rapdordata.current_embedding
    raw_diff = rapdordata.raw_lfc
    highlight_color = colors[-1] if highlight_color is None else highlight_color
    displayed_text = rapdordata.df["RAPDORid"] if sel_column is None else rapdordata.df[sel_column]
    fig = make_subplots(rows=2, cols=1, row_width=[0.85, 0.15], vertical_spacing=0.0)
    hovertext = rapdordata.df.index.astype(str) + ": " + rapdordata.df["RAPDORid"].astype(str) + "<br>Raw Log2FC: " + np.around(raw_diff, decimals=2).astype(str)
    if sel_column:
        if pd.api.types.is_float_dtype(rapdordata.df[f"{sel_column}"]):
            add =  np.around(rapdordata.df[f"{sel_column}"], decimals=2).astype(str)
        else:
            add =  rapdordata.df[f"{sel_column}"].astype(str)
        hovertext = hovertext + f"<br>{sel_column}: " + add
    clusters = np.full(embedding.shape[0], -1) if clusters is None else clusters
    n_cluster = int(np.nanmax(clusters)) + 1
    mask = np.ones(embedding.shape[0], dtype=bool)
    cutoff_mask = np.zeros(embedding.shape[0], dtype=bool)
    data = rapdordata.df["Mean Distance"]
    desired_min = 1
    min_data, max_data = np.nanmin(data), np.nanmax(data)
    marker_size = desired_min + (data - min_data) * (marker_max_size - desired_min) / (max_data - min_data)
    marker_size[np.isnan(marker_size)] = 1
    fig.add_shape(type="rect",
                  x0=-2, y0=-2, x1=2, y1=2,
                  fillcolor=second_bg_color,
                  layer="below"
                  )
    circles = np.asarray([0.3, 0.6, 1.]) * max_data
    legend_marker_sizes = desired_min + (circles - min_data) * (marker_max_size - desired_min) / (max_data - min_data)
    xloc = [legend_start + idx * legend_spread for idx in range(3)]
    fig.add_trace(
        go.Scatter(
            x=xloc,
            y=np.full(len(xloc), 0.5),
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", line=dict(color=bubble_legend_color, width=1),
                        size=legend_marker_sizes),
            name=f"Size 100",
            showlegend=False,
            hoverinfo='skip',

        ),
        row=1,
        col=1

    )
    for idx, entry in enumerate(circles):
        fig.add_annotation(
            xref="x",
            yref="y",
            xanchor="left",
            yanchor="middle",
            x=xloc[idx] + 0.02 + idx * 0.02 / 3,
            y=0.5,
            text=f"{entry:.1f}",
            showarrow=False,
            font=(dict(size=16)),
            row=1,
            col=1
        )
    fig.add_annotation(
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="middle",
        x=0.01,
        y=0.5,
        text="Mean Distance",
        showarrow=False,
        font=(dict(size=18)),
        row=1,
        col=1
    )
    if cutoff_range is not None:
        assert cutoff_type is not None
        indices = rapdordata.df[
            (rapdordata.df[cutoff_type] <= cutoff_range[1]) & (rapdordata.df[cutoff_type] >= cutoff_range[0])].index
        cutoff_mask[indices] = 1
    else:
        cutoff_mask = np.ones(embedding.shape[0], dtype=bool)

    if highlight is not None and len(highlight) > 0:
        indices = np.asarray([rapdordata.df.index.get_loc(idx) for idx in highlight])
        mask[indices] = 0
    colorscale = [colors[-1], colors[-2], colors[-1]]
    cmin = -1
    cmax = 1
    ccscale = dict(cmin=cmin, cmax=cmax, colorscale=colorscale)
    if n_cluster > len(colors) - 2:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="middle",
            x=0.5,
            y=0.5,
            text="Too Many Clusters<br> Will not show all<br>Please adjust cluster Settings",
            showarrow=False,
            font=(dict(size=28)),
            row=2,
            col=1
        )

    if np.any(clusters == -1):
        c_mask = mask & (clusters == -1) & cutoff_mask
        marker_color = raw_diff[c_mask] if show_lfc else colors[-2]
        fig.add_trace(go.Scatter(
            x=embedding[c_mask, :][:, 0],
            y=embedding[c_mask, :][:, 1],
            mode="markers",
            hovertext=hovertext[c_mask],
            marker=dict(color=marker_color, size=marker_size[c_mask],
                        colorbar=dict(
                            thickness=20,
                            nticks=3,
                            dtick=1,
                            title="Raw Log2FC",
                            tick0=0,
                            ticktext=[u"\u2264 -1", "0", u"\u2265 1"],
                            tickvals=[-1, 0, 1],
                            tickmode="array"

                        ),
                        **ccscale,),
            name=f"Not Clustered",

        ),
            row=2,
            col=1
        )
        nmask = ~mask & (clusters == -1) & cutoff_mask
        texts = displayed_text[nmask]
        marker_color = raw_diff[nmask] if show_lfc else colors[-2]
        embx = embedding[nmask, :][:, 0]
        emby = embedding[nmask, :][:, 1]
        for idx, text in enumerate(texts):
            if not pd.isna(text):
                fig.add_annotation(
                    text=text,
                    x=embx[idx],
                    y=emby[idx],
                    xanchor="center",
                    yanchor="middle",
                    showarrow=False,
                    xref="x2",
                    yref="y2",

                )

        fig.add_trace(
            go.Scatter(
                x=embx,
                y=emby,
                mode="markers",
                hovertext=hovertext[nmask],
                marker=dict(color=marker_color, size=marker_size[nmask], line=dict(color=highlight_color, width=4), **ccscale),
                name="Not Clustered",

            ),
            row=2,
            col=1
        )
    for color_idx, cluster in enumerate(range(min(n_cluster, len(colors) - 2))):
        c_mask = mask & (clusters == cluster) & cutoff_mask
        marker_color = raw_diff[c_mask] if show_lfc else colors[color_idx]


        fig.add_trace(go.Scatter(
            x=embedding[c_mask, :][:, 0],
            y=embedding[c_mask, :][:, 1],
            mode="markers",
            hovertext=hovertext[c_mask],
            marker=dict(color=marker_color, size=marker_size[c_mask], **ccscale),
            name=f"Cluster {cluster}",

        ),
            row=2,
            col=1
        )
        nmask = ~mask & (clusters == cluster) & cutoff_mask
        marker_color = raw_diff[nmask] if show_lfc else colors[color_idx]

        fig.add_trace(
            go.Scatter(
                x=embedding[nmask, :][:, 0],
                y=embedding[nmask, :][:, 1],
                mode="markers",
                hovertext=hovertext[nmask],
                marker=dict(color=marker_color, size=marker_size[nmask], line=dict(color=highlight_color, width=4), **ccscale),
                name=f"Cluster {cluster}",
            ),
            row=2,
            col=1
        )
    fig.update_layout(
        legend=dict(
            title="Clusters",
            yanchor="top",
            yref="paper",
            y=0.85,

        ),
        margin=dict(r=0, l=0)
    )
    fig.update_layout(
        xaxis2=dict(title="relative fraction shift"),
        yaxis2=dict(title="relative distribution change"),
        yaxis=dict(range=[0, 1], showgrid=False, showline=False, showticklabels=False, zeroline=False, ticklen=0,
                   fixedrange=True),
        xaxis=dict(range=[0, 1], showgrid=False, showline=False, showticklabels=False, zeroline=False, ticklen=0,
                   fixedrange=True),
        legend={'itemsizing': 'constant'},
        showlegend=not show_lfc,


    )
    if not show_lfc:
        fig.update_traces(marker=dict(colorbar=None, colorscale=None, showscale=False))
        fig.update_layout(coloraxis_colorbar=None, coloraxis=None, coloraxis_showscale=False)
    fig.update_xaxes(categoryorder='array', categoryarray=rapdordata.fractions)
    return fig

def plot_distance_and_var(rapdordata: RAPDORData, colors, var_type: str = "ANOSIM R", title_col: str = "RAPDORid", highlight = None, show_lfc: bool = False):
    fig = go.Figure()
    mask = np.ones(rapdordata.df.shape[0], dtype=bool)
    hovertext = rapdordata.df.index.astype(str) + ": " + rapdordata.df["RAPDORid"].astype(str)
    raw_diff = rapdordata.raw_lfc

    if title_col:
        if pd.api.types.is_float_dtype(rapdordata.df[f"{title_col}"]):
            add = np.around(rapdordata.df[f"{title_col}"], decimals=2).astype(str)
        else:
            add = rapdordata.df[f"{title_col}"].astype(str)
        hovertext = hovertext + f"<br>{title_col}: " + add
    if highlight is not None and len(highlight) > 0:
        highlight = rapdordata[highlight]

        indices = np.asarray([rapdordata.df.index.get_loc(idx) for idx in highlight])
        mask[indices] = 0
    y = rapdordata.df[var_type]
    if show_lfc:
        hovertext = hovertext + "<br>Raw Log2FC: " + np.around(raw_diff, decimals=2).astype(str)
    if "p-Value" in var_type:
        y = -1 * np.log10(y)
        var_type = f"-log<sub>10</sub>({var_type})"
    colorscale = [colors[-1], colors[-2], colors[-1]]
    cmin = -1
    cmax = 1
    ccscale = dict(cmin=cmin, cmax=cmax, colorscale=colorscale)


    fig.add_trace(
        go.Scatter(
            y=y[mask],
            x=rapdordata.df[mask]["Mean Distance"],
            mode="markers",
            hovertext=hovertext[mask],
            marker=dict(
                color=raw_diff[mask] if show_lfc else colors[0],
                colorbar=dict(
                    thickness=20,
                    nticks=3,
                    dtick=1,
                    title="Raw Log2FC",
                    tick0=0,
                    ticktext=[u"\u2264 -1", "0", u"\u2265 1"],
                    tickvals=[-1, 0, 1],
                    tickmode="array"

                ),
                **ccscale
            ),
            showlegend=False,

        ),
    )
    fig.add_trace(
        go.Scatter(
            y=y[~mask],
            x=rapdordata.df[~mask]["Mean Distance"],
            mode="markers",
            name="highlighted",
            showlegend=False,
            hovertext=hovertext[~mask],
            marker=dict(color=raw_diff[mask] if show_lfc else colors[0], line=dict(width=4, color=colors[1]), **ccscale)
        ),
    )
    texts = rapdordata.df[~mask][title_col]
    embx = rapdordata.df[~mask]["Mean Distance"]
    emby = y[~mask]
    for idx, text in enumerate(texts):
        if not pd.isna(text):
            fig.add_annotation(
                text=text,
                x=embx.iloc[idx],
                y=emby.iloc[idx],
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                xref="x",
                yref="y",

            )
    fig.update_xaxes(title="Mean Distance")
    fig.update_yaxes(title=var_type)
    if not show_lfc:
        fig.update_traces(marker=dict(colorbar=None, colorscale=None, showscale=False))
        fig.update_layout(coloraxis_colorbar=None, coloraxis=None, coloraxis_showscale=False)
    else:
        fig.update_layout(
            showlegend=show_lfc,
            #coloraxis_colorbar=True,
            #coloraxis=True,
            coloraxis_showscale=True
        )
    fig.update_xaxes(categoryorder='array', categoryarray=rapdordata.fractions)


    return fig

def _update_sample_histo_layout(fig, rapdordata, colors, column_titles, row_titles, y_0, x_0):
    fig.add_trace(
        go.Bar(
            x=[np.nan],
            y=[np.nan],
            showlegend=True,
            name="Same Treatment",
            marker=dict(color=colors[0])
        )
    )
    fig.add_trace(
        go.Bar(
            x=[np.nan],
            y=[np.nan],
            showlegend=True,
            name="Different Treatment",
            marker=dict(color=colors[1])
        )
    )
    fig.add_trace(
        go.Scatter(
            mode="lines",
            x=[np.nan],
            y=[np.nan],
            showlegend=True,
            name="Median",
            line=dict(width=1, dash="dot", color="black")
        )
    )
    r_y = {treatment: [] for treatment in rapdordata.treatment_levels}
    c_x = {treatment: [] for treatment in rapdordata.treatment_levels}

    for annotation in fig.layout.annotations:
        if annotation.text in column_titles:
            text, treatment = annotation.text.split("#-")
            c_x[treatment].append(annotation.x)

            annotation.update(
                y=y_0,
                text=text[1:],
                yanchor="top"
            )

        elif annotation.text in row_titles:
            text, treatment = annotation.text.split("#-")
            r_y[treatment].append(annotation.y)
            annotation.update(
                x=x_0,
                text=text[1:],
                xanchor="right"
            )
        else:
            pass
    for treatment, y_s in r_y.items():
        fig.add_annotation(
            text=treatment,
            x=x_0,
            y=np.mean(y_s),
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="middle",
            textangle=90,
            showarrow=False,
        )
        fig.add_shape(type="line",
                      x0=x_0, y0=y_s[0], x1=x_0, y1=y_s[-1], xref="paper", yref="paper", line_color="black"
                      )

    for treatment, x_s in c_x.items():
        fig.add_annotation(
            text=treatment,
            x=np.mean(x_s),
            y=y_0,
            xref="paper",
            yref="paper",
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
        )
        fig.add_shape(type="line",
                      x0=x_s[0], y0=y_0, x1=x_s[-1], y1=y_0, xref="paper", yref="paper", line_color="black"
                      )

    fig.update_layout(
        legend=dict(
            x=1, y=1, yanchor="top", xanchor="right", xref="paper", yref="paper",  # itemsizing="constant"
        )
    )
    fig.update_xaxes(showgrid=True)
    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=624,
        height=400,
    )
    return fig


def _sample_spearman(rapdordata: RAPDORData, colors: Iterable[str], x_0=1.1, y_0=1.15, **kwargs):
    names = [f"{row['Replicate']}#-{row['Treatment']}" for idx, row in rapdordata.internal_design_matrix.iterrows()]
    column_titles = [f"c{name}" for name in names[:-1]]
    row_titles = [f"r{name}" for name in names[1:]]
    defaults = {
        "vertical_spacing": 0.05,
        "horizontal_spacing": 0.03,
        "shared_xaxes": True,
        "shared_yaxes": True,
        "x_title": "Spearman R",
        "y_title": f"# of {rapdordata.measure_type}s"
    }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    array = rapdordata.kernel_array
    #array = array.reshape(array.shape[1], array.shape[0], -1)
    p, n, f = array.shape
    fig = make_subplots(rows=n - 1, cols=n - 1,
                        column_titles=column_titles, row_titles=row_titles, **kwargs
                        )

    for i in range(n):
        for j in range(i + 1, n):
            spears = np.empty(p)
            for protein in range(p):
                sub_i = array[protein][i]
                sub_j = array[protein][j]
                if np.any(np.isnan(sub_i) | np.isnan(sub_j)):
                    pearson_coefficient = np.nan
                else:
                    pearson_coefficient, _ = spearmanr(sub_i, sub_j)
                # pearson_coefficient = data.distances[protein][i][j]
                spears[protein] = pearson_coefficient
            same = rapdordata.internal_design_matrix.iloc[i]["Treatment"] == rapdordata.internal_design_matrix.iloc[j][
                "Treatment"]
            fig.add_trace(
                go.Histogram(
                    x=spears, showlegend=False,
                    marker=dict(color=colors[0] if same else colors[1]),
                    xbins=dict(start=-1, end=1, size=0.01)
                ), row=j, col=i + 1
            )
            m = np.nanmedian(spears)
            fig.add_vline(
                x=m,
                line_color="black",
                line_width=1,
                row=j,
                col=i + 1,
                line_dash="dot",
            )
    fig = _update_sample_histo_layout(fig, rapdordata, colors, column_titles, row_titles, y_0, x_0)
    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(autorange=True)
    return fig


def _sample_jsd_histograms(rapdordata: RAPDORData, colors: Iterable[str], x_0=1.1, y_0=1.15, **kwargs):
    names = [f"{row['Replicate']}#-{row['Treatment']}" for idx, row in rapdordata.internal_design_matrix.iterrows()]
    column_titles = [f"c{name}" for name in names[:-1]]
    row_titles = [f"r{name}" for name in names[1:]]
    defaults = {
        "vertical_spacing": 0.05,
        "horizontal_spacing": 0.03,
        "shared_xaxes": "all",
        "shared_yaxes": "all",
        "x_title": "Jensen-Shannon-Distance",
        "y_title": f"# of {rapdordata.measure_type}s"
    }
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
    p, n, f = rapdordata.norm_array.shape
    fig = make_subplots(rows=n - 1, cols=n - 1,
                        column_titles=column_titles, row_titles=row_titles, **kwargs
                        )
    for i in range(n):
        for j in range(i + 1, n):
            jsds = np.empty(p)
            for protein in range(p):
                jsd = rapdordata.distances[protein][i][j]
                jsds[protein] = jsd
            same = rapdordata.internal_design_matrix.iloc[i]["Treatment"] == rapdordata.internal_design_matrix.iloc[j][
                "Treatment"]
            fig.add_trace(
                go.Histogram(
                    x=jsds, showlegend=False,
                    marker=dict(color=colors[0] if same else colors[1]),
                    xbins=dict(start=0, end=1, size=0.01)
                ), row=j, col=i + 1
            )
            m = np.nanmedian(jsds)
            fig.add_vline(
                x=m,
                line_color="black",
                line_width=1,
                row=j,
                col=i + 1,
                line_dash="dot",
            )
    fig = _update_sample_histo_layout(fig, rapdordata, colors, column_titles, row_titles, y_0, x_0)
    return fig


def plot_sample_histogram(rapdordata: RAPDORData, method: str = "spearman",
                          colors: Iterable[str] = COLOR_SCHEMES["Flamingo"], **kwargs):
    """ Plots the distribution of jensen-shannon-distance/spearman R for all pairwise samples

    Args:
        rapdordata (RAPDORData): A :class:`~RAPDOR.datastructures.RAPDORData` object
        method: either spearman or jensen-shannon-distance (jsd).
        colors: colors to use for plotting
        **kwargs: will be passed to the plotly make_subplots call

    Returns:

    """
    if method == "spearman":
        return _sample_spearman(rapdordata, colors, **kwargs)
    elif method == "jensen-shannon-distance" or method == "jsd":
        return _sample_jsd_histograms(rapdordata, colors, **kwargs)
    else:
        raise ValueError(f"Mode {method} not supported")


def _get_x(rapdordata, ntop, use_raw, summarize_fractions):
    x = rapdordata.array if use_raw else rapdordata.norm_array
    if summarize_fractions:
        x = np.transpose(x, (1, 2, 0))
        x = x.reshape(x.shape[0], -1).T
    else:
        x = x.reshape(x.shape[0], -1)
    ntop = x.shape[0] if ntop is None else ntop
    if isinstance(ntop, float):
        if not (0 <= ntop <= 1):
            raise ValueError("ntop must be an integer or float between 0 and 1")
        ntop = int(x.shape[0] * ntop)
    var = np.var(x, axis=1)
    x = x[~np.isnan(var)]
    var = var[~np.isnan(var)]
    top_indices = np.argsort(var)[-ntop:]
    x = x[top_indices]
    return x

def plot_ep_vs_ep(rapdordata, use_raw: bool = False, colors: Tuple = COLOR_SCHEMES["Flamingo"], x_0=1.1, y_0=1.15, **kwargs):
    x =rapdordata.norm_array
    names = [f"{row['Replicate']}#-{row['Treatment']}" for idx, row in rapdordata.internal_design_matrix.iterrows()]
    column_titles = [f"c{name}" for name in names[:-1]]
    row_titles = [f"r{name}" for name in names[1:]]
    mask = np.all(np.isnan(x), axis=-1)
    i = rapdordata.state.kernel_size // 2
    positions = np.asarray(rapdordata.fractions)
    if i != 0:
        positions = positions[i:-i]
    x = (positions * x).sum(axis=-1)
    p, n, f = rapdordata.norm_array.shape

    fig = make_subplots(rows=n, cols=n,
                        column_titles=column_titles, row_titles=row_titles, **kwargs
                        )

    hover = rapdordata.df["RAPDORid"].astype(str)
    c = x[~np.any(np.isnan(x), axis=1)]
    corrs = np.corrcoef(c.T)
    for i in range(n):
        for j in range(i + 1, n):
            if i == 0 and j == 1:
                p = 0
            m = mask[:, i] | mask[:, j]
            same = rapdordata.internal_design_matrix.iloc[i]["Treatment"] == rapdordata.internal_design_matrix.iloc[j][
                "Treatment"]
            fig.add_trace(
                go.Scatter(
                    x=x[:, i][~m],
                    y=x[:, j][~m],
                    showlegend=False,
                    marker=dict(color=colors[0] if same else colors[1], size=2),
                    mode="markers",
                    hovertext=hover[~m]
                ), row=j+1, col=i + 1
            )
            fig.add_trace(
                go.Heatmap(
                    x=[1],
                    y=[1],
                    z=[corrs[i, j]],
                    colorscale=[colors[0], "white", colors[1]],
                    zmin=-1,
                    zmax=1,

                ),
                row=i + 1, col=j + 1
            )
            fig.add_annotation(
                text=f"{corrs[i, j]:.3f}",
                xanchor="center",
                yanchor="middle",
                x=1,
                y=1,
                showarrow=False,
                row=i + 1, col=j + 1

            )
            fig.update_yaxes(showticklabels=False, showline=False, showgrid=False, row=i+1, col=j+1)
            fig.update_xaxes(showticklabels=False, showline=False, showgrid=False, row=i+1, col=j+1)

    fig = _update_sample_histo_layout(fig, rapdordata, colors, column_titles, row_titles, y_0, x_0)
    fig.update_xaxes(showticklabels=False, showline=False, showgrid=False, row=1, col=1, zeroline=False)
    fig.update_yaxes(showticklabels=False, showline=False, showgrid=False, row=1, col=1, zeroline=False)
    fig.update_layout(
        legend=dict(
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom"
        )
    )
    fig.data = fig.data[:-1]
    return fig



def _get_sorted_design(rapdordata, summarize_fractions, x):
    sorted_df = rapdordata.internal_design_matrix.sort_values("index")
    if not summarize_fractions:
        if not rapdordata.categorical_fraction and rapdordata.state.kernel_size != 0:
            i = int((rapdordata.state.kernel_size - 1) / 2)
            fractions = rapdordata.fractions[i:-i]
        else:
            fractions = rapdordata.fractions
        sorted_df = sorted_df.merge(pd.Series(fractions, name="Fraction"), how="cross")
        sorted_df["displayName"] = sorted_df["Treatment"].astype(str) + " - " + sorted_df["Fraction"].astype(str) + " - " + \
                                sorted_df["Replicate"].astype(str)
        sorted_df = sorted_df.sort_values(["Treatment", "Fraction", "Replicate"])
        x = x[: ,sorted_df.index]
        sorted_df["legendGroup"] = sorted_df["Treatment"].astype(str) + "-" + sorted_df["Fraction"].astype(str)


    else:
        sorted_df["displayName"] = sorted_df["Treatment"].astype(str) + " - " + sorted_df["Replicate"].astype(str)
        sorted_df["legendGroup"] = sorted_df["Treatment"].astype(str)

    return sorted_df, x

def _get_x_and_sorted_design(rapdordata, ntop, use_raw, summarize_fractions):
    x = _get_x(rapdordata, ntop, use_raw, summarize_fractions)
    sorted_df, x = _get_sorted_design(rapdordata, summarize_fractions, x)
    return sorted_df, x


def _2d_pca(plot_dims, pca, sorted_df, colors, summarize_fractions):
    fig = go.Figure()
    symbols = SYMBOLS
    marker_colors = {treatment: colors[idx] for idx, treatment in enumerate(sorted_df["Treatment"].unique())}
    shape_col = "Replicate" if summarize_fractions else "Fraction"
    marker_shapes = {frac: symbols[idx] for idx, frac in enumerate(sorted_df[shape_col].unique())}
    for idx, row in sorted_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[f"PCA{plot_dims[0]}"]],
                y=[row[f"PCA{plot_dims[1]}"]],
                name=row["displayName"] if not summarize_fractions else row["Replicate"],
                legendgroup=row["legendGroup"],
                legendgrouptitle=dict(text=row["legendGroup"]),
                marker=dict(color=marker_colors[row["Treatment"]], size=10, symbol=marker_shapes[row[shape_col]]),
                mode="markers"
            )
        )
    fig.update_xaxes(
        title=f"PC{plot_dims[0]} ({pca.explained_variance_ratio_[plot_dims[0]] * 100:.2f}%)"
    )
    fig.update_yaxes(
        title=f"PC{plot_dims[1]} ({pca.explained_variance_ratio_[plot_dims[1]] * 100:.2f}%)"
    )
    return fig


def _3d_pca(plot_dims, pca, sorted_df, colors, summarize_fractions ):
    symbols = SYMBOLS3D
    shape_col = "Replicate" if summarize_fractions else "Fraction"

    fig = go.Figure()
    marker_colors = {treatment: colors[idx] for idx, treatment in enumerate(sorted_df["Treatment"].unique())}
    while len(symbols) < len(sorted_df[shape_col].unique()):
        symbols = symbols + symbols
    marker_shapes = {frac: symbols[idx] for idx, frac in enumerate(sorted_df[shape_col].unique())}

    for idx, row in sorted_df.iterrows():
        fig.add_trace(
            go.Scatter3d(
                x=[row[f"PCA{plot_dims[0]}"]],
                y=[row[f"PCA{plot_dims[1]}"]],
                z=[row[f"PCA{plot_dims[2]}"]],
                name=row["displayName"],
                legendgroup=row["legendGroup"],
                legendgrouptitle=dict(text=row["legendGroup"]),
                marker=dict(color=marker_colors[row["Treatment"]], size=5, symbol=marker_shapes[row[shape_col]]),
                mode="markers"
            )
        )
    fig.update_layout(
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(
                title=f"PC{plot_dims[0]} ({pca.explained_variance_ratio_[plot_dims[0]] * 100:.2f}%)",
            ),
            yaxis=go.layout.scene.YAxis(
                title=f"PC{plot_dims[1]} ({pca.explained_variance_ratio_[plot_dims[1]] * 100:.2f}%)",

            ),
            zaxis=go.layout.scene.ZAxis(
                title=f"PC{plot_dims[2]} ({pca.explained_variance_ratio_[plot_dims[2]] * 100:.2f}%)",

            ),
            aspectratio=dict(x=1, y=1, z=1)
        )

    )
    return fig


def plot_sample_pca(
        rapdordata: RAPDORData,
        plot_dims: Union[Tuple[int, int], Tuple[int, int, int]],
        ntop=None,
        summarize_fractions: bool = True,
        use_raw: bool = False,
        colors = None,

):
    """Creates PCA plot of the samples of an RAPDORdata object.

    It can either produce a 3D or 2D PCA plot depending on the number of dimensions specified in plot_dims
    If summarize_fractions is True it will flatten the fraction dimension.
    If it is set to false it will treat each replicate, treatment, fraction combination as a separate sample.
    Args:
        rapdordata (RAPDORData): A :class:`~RAPDOR.datastructures.RAPDORData` object
        plot_dims (Union[Tuple[int, int], Tuple[int, int, int]]): The principal components to plot. Either a Tuple of
            three or two components. Three results in a 3D and two in a 2D plot.
        ntop (int or float): use the n top entries regarding their variance if its an int. If it is a float it uses that
            percentage of the data.
        use_raw (bool): uses raw values instead of normalized values.
        summarize_fractions (bool): flattens the fractions if True only displaying one point per sample and treatment.
        colors (Iterable[str]): Iterable of color values

    Returns: go.Figure()
    """
    if colors is None:
        colors = list(DEFAULT_COLORS.values())

    sorted, x = _get_x_and_sorted_design(rapdordata, ntop, use_raw, summarize_fractions)

    pca = PCA(n_components=None)

    principal_components = pca.fit_transform(x.T)
    columns = {f"PCA{idx + 1}": principal_components[:, idx] for idx in range(principal_components.shape[1])}
    sorted = pd.concat([sorted, pd.DataFrame(columns)], axis=1)

    if len(plot_dims) == 2:
        fig = _2d_pca(plot_dims, pca, sorted, colors, summarize_fractions)

    elif len(plot_dims) == 3:
        fig = _3d_pca(plot_dims, pca, sorted, colors, summarize_fractions)
    else:
        raise ValueError("Can only plot between 2 and 3 dimensions for PCA please adjust plot_dims")

    fig.update_layout(
        legend=dict(groupclick="toggleitem")

    )
    return fig


def plot_distance_stats(
        rapdordata: RAPDORData,
        colors: Iterable = COLOR_SCHEMES["Flamingo"]

):
    fig = make_subplots(2, 2, row_titles=rapdordata.treatment_levels, column_titles=["Mean", "Var"])
    for i, treatment in enumerate(rapdordata.treatment_levels):
        x = rapdordata.df[f"{treatment} distance mean"]
        fig.add_trace(
            go.Histogram(
                x=x,
                marker=dict(color=colors[i])
            ),
            row=i+1,
            col=1
        )
        fig.add_trace(
            go.Histogram(
                x=rapdordata.df[f"{treatment} distance var"],
                marker=dict(color=colors[i])

            ),
            row=i+1,
            col=2
        )
        perc = np.nanpercentile(x, 95)
        fig.add_vline(x= perc, row=i+1, col=1, annotation_text=f"{perc:.3f}")
    return fig


def plot_sample_correlation(
        rapdordata: RAPDORData,
        ntop: Union[int, float] = None,
        use_raw: bool = False,
        summarize_fractions: bool = True,
        method: str = "pearson",
        colors: Iterable[str] = None,
        highlight_replicates: bool = False,
        show_values: bool = False
):
    """Creates a heatmap of correlations of different samples.
    If summarize_fractions is True it will flatten the fraction dimension.
    If it is set to false it will treat each replicate, treatment, fraction combination as a separate sample.

    Args:
        rapdordata (RAPDORData): A :class:`~RAPDOR.datastructures.RAPDORData` object
        ntop (int or float): use the n top entries regarding their variance if it is an int. If it is a float it uses that
            percentage of the data.
        use_raw (bool): uses raw values instead of normalized values.
        summarize_fractions (bool): Creates correlation heatmap using only samples and flattens the fractions if True.
        method (str): One of spearman or pearson
        colors (Iterable[str]): Iterable of color values
        highlight_replicates: if True will but a box around replicates belonging to the same treatment. Only works with
            summarize_fractions set to false

    Returns: go.Figure()


    """
    if colors is None:
        colors = list(DEFAULT_COLORS.values())
        colors = [colors[0], "white", colors[1]]
    fig = go.Figure()
    sorted_df, x = _get_x_and_sorted_design(rapdordata, ntop, use_raw, summarize_fractions)
    if method == "pearson":
        x = x[~np.any(np.isnan(x), axis=1)]
        res = np.corrcoef(x.T)
    elif method == "spearman":
        res, pvals = spearmanr(x, nan_policy="omit")
    else:
        raise ValueError("Unupported method. Must be one of pearson or spearman")
    fig.add_trace(
        go.Heatmap(
            z=res,
            x=sorted_df["displayName"],
            y=sorted_df["displayName"],
            colorscale=colors,
            zmin=-1,
            zmax=1,
            text=np.round(res, 2) if show_values else None,
            texttemplate="%{text}" if show_values else None,
            textfont=dict(size=12, color="black")

        )
    )
    if highlight_replicates:
        sorted_df["sortIDX"] = np.arange(sorted_df.shape[0])
        indices = sorted_df.groupby(["Treatment", "Fraction"])["sortIDX"].aggregate(["min", "max"])
        for _, row in indices.iterrows():
            fig.add_shape(
                type="rect",
                x0=row["min"]-0.5,
                x1=row["max"]+0.5,
                y0=row["min"]-0.5,
                y1=row["max"]+0.5

            )
    if not summarize_fractions:

        sorted_df["sortIDX"] = np.arange(sorted_df.shape[0])
        xrange = [sorted_df["sortIDX"].min(), sorted_df["sortIDX"].max()]
        fig.update_xaxes(
            tickvals=list(range(*xrange)),
            ticktext=sorted_df["Replicate"],
            tickmode="array",
            range=xrange
        )
        fig.update_yaxes(
            showticklabels=False,
            range=xrange
        )
        indices = sorted_df.groupby(["Treatment"])["sortIDX"].aggregate(["min", "max"])
        for _, row in indices.iterrows():
            x = -0.05
            scale_factor = 1
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=row["min"],
                y1=row["max"],
                xref="paper"
            )
            fig.add_shape(
                type="line",
                x0=row["min"],
                x1=row["max"],
                y0=1-x * scale_factor,
                y1=1-x * scale_factor,
                yref="paper"
            )
            fig.add_annotation(
                text=row.name,
                x=x,
                y=(row["min"] + row["max"]) / 2,
                xanchor="right",
                yanchor="middle",
                xref="paper",
                showarrow=False,
                textangle=-90
            )
            fig.add_annotation(
                text=row.name,
                y=1-x * scale_factor,
                x=(row["min"] + row["max"]) / 2,
                yanchor="bottom",
                xanchor="center",
                yref="paper",
                showarrow=False,
            )
        indices = sorted_df.groupby(["Treatment", "Fraction"])["sortIDX"].aggregate(["min", "max"])
        for _, row in indices.iterrows():
            x = -0.01
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                y0=row["min"]-0.25,
                y1=row["max"]+0.25,
                xref="paper"
            )
            fig.add_shape(
                type="line",
                y0=1-x,
                y1=1-x,
                x0=row["min"] - 0.25,
                x1=row["max"] + 0.25,
                yref="paper"
            )
            fig.add_annotation(
                text=row.name[-1],
                x=x,
                y=(row["min"] + row["max"]) / 2,
                yanchor="middle",
                xanchor="right",
                xref="paper",
                showarrow=False,
            )
            fig.add_annotation(
                text=row.name[-1],
                y=1-x ,
                x=(row["min"] + row["max"]) / 2,
                yanchor="bottom",
                xanchor="center",
                yref="paper",
                showarrow=False,
                textangle=-90
            )

    fig.update_layout(
        template=DEFAULT_TEMPLATE,
        width=624,
        height=624,
        margin=dict(r=50, b=50)
    )
    return fig


def plot_sum_of_intensities(rapdordata, colors: Iterable = COLOR_SCHEMES["Flamingo"], normalize: bool = False, show_last: bool = True):
    """
        Plots a heatmap of summed intensities across fractions and replicates from RAPDOR data.

        The function sums intensity values across all proteins (or features) for each fraction and replicate,
        optionally normalizes the sums per replicate, and visualizes the results as a heatmap.

        Args:
            rapdordata:
                An object containing RAPDOR data with the following attributes:
                - `.array`: 2D numpy array of intensities with shape (samples, fractions).
                - `.fractions`: List or array of fraction identifiers.
                - `.internal_design_matrix`: DataFrame with columns `"Treatment"` and `"Replicate"`.
            colors (Iterable, optional):
                A color scale iterable for the heatmap (e.g., a Plotly colorscale). Default is COLOR_SCHEMES["Flamingo"].
            normalize (bool, optional):
                Whether to normalize intensities within each replicate so that sums per replicate equal 1. Defaults to False.
            show_last (bool, optional):
                Whether to include the last fraction in the heatmap. Defaults to True.

        Returns:
            plotly.graph_objects.Figure:
                A Plotly heatmap figure showing summed (or normalized) intensities by fraction and replicate.

        Example:
            fig = plot_sum_of_intensities(rapdordata, normalize=True)
            fig.show()

    """
    fig = go.Figure()
    z = np.nansum(rapdordata.array, axis=0)
    title = "Intensity"
    if normalize:
        z = z / z.sum(axis=1, keepdims=True)
        title = "Normalized intensity"
    fractions = rapdordata.fractions

    if not show_last:
        z = z[:, :-1]
        fractions = fractions[:-1]
    names = (rapdordata.internal_design_matrix["Treatment"].astype(str) + " " + rapdordata.internal_design_matrix["Replicate"].astype(str)).tolist()
    fig.add_trace(
        go.Heatmap(
            x=fractions,
            y=names,
            z=z,
            colorscale=colors,
            colorbar=dict(
                title=title,
            )
        )
    )
    fig.update_xaxes(title="Fraction")
    fig.update_yaxes(title="Replicate")

    return fig


if __name__ == '__main__':
    from RAPDOR.datastructures import RAPDORData

    df = pd.read_csv("tests/testData/testFile.tsv", sep="\t")
    df = pd.read_csv("../testData/sanitized_df.tsv", sep="\t")
    design = pd.read_csv("tests/testData/testDesign.tsv", sep="\t")
    design = pd.read_csv("../testData/sanitized_design.tsv", sep="\t")
    rapdor = RAPDORData(df, design, logbase=2, control="CTRL")
    dolphin = list(COLOR_SCHEMES["Dolphin"])
    dolphin.insert(1, "white")
    rapdor.normalize_array_with_kernel(kernel_size=0)
    rapdor.calc_distances()
    rapdor = RAPDORData.from_file("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/RAPDORAnalysis/GradRData.json")
    #rapdor = RAPDORData.from_file("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/RAPDORonRDeeP/RDeePRAPDOR.json")

    rapdor.df["RAPDORid"] = rapdor.df.index
    #fig = plot_sample_correlation(rapdor, method="spearman", show_values=True, summarize_fractions=True, use_raw=False, highlight_replicates=False, ntop=None, colors=dolphin)
    fig = plot_sum_of_intensities(rapdor, dolphin, normalize=False, show_last=True)
    fig.write_image("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/SumOfIntensities.svg")

    fig = plot_sum_of_intensities(rapdor, dolphin, normalize=True, show_last=True)
    fig.write_image("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/SumOfIntensitiesNormalized.svg")
    fig.show()
    exit()
    #rapdor = RAPDORData.from_file("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/RAPDORonRDeeP/RDeePRAPDOR.json")
    #fig = plot_sample_pca(rapdor, plot_dims=(1, 2, 3), ntop=0.2, colors=COLOR_SCHEMES["Dolphin"], use_raw=False, summarize_fractions=False)
    fig.show()
    #fig = plot_sample_histogram(rapdordata=rapdor, method="spearman")
    #fig.show()
    exit()
    rapdor.calc_distances(method="Jensen-Shannon-Distance")
    rapdor.calc_all_scores()
    rapdor.calc_distribution_features()
    rapdor.rank_table(["ANOSIM R", "Mean Distance"], ascending=[False, False])
    rapdor = RAPDORData.from_file("/home/rabsch/PythonProjects/synRDPMSpec/Pipeline/NatureMouse/mobilityScore/mobilityegf_2min.json")
    subdata = rapdor.norm_array[0]
    fig = plot_protein_distributions([rapdor.df.iloc[0]["RAPDORid"]], rapdor, colors=COLOR_SCHEMES["Dolphin"], mode="bar",barmode="f")

    fig.update_layout(width=600)
    fig.show()
    exit()
    fig = plot_distance_and_var(rapdor, colors=COLOR_SCHEMES["Dolphin"], show_lfc=True)
    fig.show()
    exit()
    print(rapdor.df["Gene"].str.contains('rpl|Rpl'))
    ids = list(rapdor.df[rapdor.df["small ribo"] == True]["RAPDORid"])[0:5]
    ids2 = list(rapdor.df[rapdor.df["large ribo"] == True]["RAPDORid"])
    ids3 = list(rapdor.df[rapdor.df["photosystem"] == True]["RAPDORid"])
    ids = []
    ids += list(rapdor.df[rapdor.df["old_locus_tag"].str.contains("sll1388|slr0711")]["RAPDORid"])
    d = {"large Ribo": ids2, "foo": ids}

    # plt = plot_dimension_reduction(rapdor, colors=COLOR_SCHEMES["Dolphin"], highlight=ids  )
    plt = rank_plot(d, rapdor, colors=COLOR_SCHEMES["Dolphin"], orientation="v", tri_y=0.1, triangles="inside")
    plt.update_xaxes(dtick=25)
    plt.show()
    exit()

    # fig = multi_means_and_histo(d, rapdor, colors=COLOR_SCHEMES["Dolphin"] + COLOR_SCHEMES["Viking"])
    # fig = plot_protein_distributions(ids[0:4], rapdor, mode="bar", plot_type="mixed", colors=COLOR_SCHEMES["Dolphin"])
    fig = plot_sample_histogram(rapdor, method="jsd")
    fig.show()
    fig = plot_sample_histogram(rapdor)
    fig.show()
    fig.write_image("foo.svg")
