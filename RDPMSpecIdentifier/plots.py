import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

DEFAULT_COLORS = [
    'rgb(138, 255, 172)', 'rgb(255, 138, 221)',
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
    'rgb(44, 160, 44)',
    'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
    'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
    'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
]

def color_to_calpha(color: str, alpha: float = 0.2):
    color = color.split("(")[-1].split(")")[0]
    return f"rgba({color}, {alpha})"


def plot_pca(components, labels, to_plot: tuple = (0, 1, 2)):
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


def plot_replicate_distribution(subdata, design: pd.DataFrame, groups: str, offset: int = 0, colors = None):
    if colors is None:
        colors = DEFAULT_COLORS
    indices = design.groupby(groups, group_keys=True).apply(lambda x: list(x.index))
    x = list(range(offset+1, subdata.shape[1] + offset+1))
    fig = go.Figure()
    names = []
    values = []
    for eidx, (name, idx) in enumerate(indices.items()):
        name = f"{groups}: {name}".ljust(15, " ")
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
                    line=dict(width=5)
                )
            )
    fig.add_traces(values)
    fig = update_distribution_layout(fig, names, x, offset)
    return fig




def plot_distribution(subdata, design: pd.DataFrame, groups: str, offset: int = 0, colors = None):
    if colors is None:
        colors = DEFAULT_COLORS
    fig = go.Figure()
    indices = design.groupby(groups, group_keys=True).apply(lambda x: list(x.index))
    medians = []
    means = []
    errors = []
    x = list(range(offset+1, subdata.shape[1] + offset+1))
    names = []
    for eidx, (name, idx) in enumerate(indices.items()):
        name = f"{groups}: {name}".ljust(15, " ")
        legend=f"legend{eidx+1}"
        names.append(name)
        median_values = np.nanmedian(subdata[idx,], axis=0)
        mean_values = np.nanmean(subdata[idx,], axis=0)
        max_values = np.nanquantile(subdata[idx,], 0.75, axis=0)
        max_values = np.nanmax(subdata[idx,], axis=0)
        min_values = np.nanquantile(subdata[idx,], 0.25, axis=0)
        min_values = np.nanmin(subdata[idx,], axis=0)
        color = colors[eidx]
        a_color = color_to_calpha(color, 0.4)
        medians.append(go.Scatter(
            x=x,
            y=median_values,
            marker=dict(color=colors[eidx]),
            name="Median",
            legend=legend,
            line=dict(width=5)
            ))
        means.append(go.Scatter(
            x=x,
            y=mean_values,
            marker=dict(color=colors[eidx]),
            name="Mean",
            legend=legend,
            line=dict(width=3, dash="dash")
        ))
        y = np.concatenate((max_values, np.flip(min_values)), axis=0)
        errors.append(
            go.Scatter(
                x=x + x[::-1],
                y=y,
                marker=dict(color=colors[eidx]),
                name="Min-Max",
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
    fig = update_distribution_layout(fig, names, x, offset)
    return fig


def update_distribution_layout(fig, names, x, offset):
    fig.update_layout(hovermode="x")
    fig.update_layout(xaxis_range=[x[0] - offset - 0.5, x[-1] + offset + 0.5])
    fig.update_layout(
        yaxis_title="Protein Amount in Fraction [%]",
    )
    fig.update_layout(
        legend=dict(
            title=names[0],
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
        ),
        legend2=dict(
            title=names[1],
            orientation="h",
            yanchor="bottom",
            y=1.2,
            xanchor="left",
            x=0,
        )
    )
    return fig


def plot_correlation_heatmap(array, gene_id, design: pd.DataFrame, df: pd.DataFrame, groups: str):
    loc = df.index.get_loc(gene_id)
    subdata = array[loc]
    names = groups + design[groups].astype(str) + " " + design["replicate"].astype(str)
    corr_coeff = np.corrcoef(subdata)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_coeff,
            x=names,
            y=names,
            colorscale="RdBu_r",

        )
    )
    return fig


def plot_heatmap(distances, design: pd.DataFrame, groups: str, colors=None):
    if colors is None:
        colors = DEFAULT_COLORS
    names = groups + design[groups].astype(str) + " " + design["Replicate"].astype(str)
    fig = go.Figure(
        data=go.Heatmap(
            z=distances[:, ::-1],
            x=names,
            y=names[::-1],
            colorscale=colors[:2]
        )
    )

    return fig

def plot_barcode_plot(subdata, design: pd.DataFrame, groups, offset: int = 0, colors=None):
    if colors is None:
        colors = DEFAULT_COLORS
    fig = go.Figure()
    indices = design.groupby(groups, group_keys=True).apply(lambda x: list(x.index))
    fig = make_subplots(cols=1, rows=2, vertical_spacing=0)

    scale = [[[0.0, "rgba(240, 40, 145, 0)"], [1, "rgba(240, 40, 145, 1)"]], [[0.0, "rgba(39, 241, 245, 0)"], [1, "rgba(39, 241, 245, 1)"]]]
    ys = []
    scale = []
    means = []
    xs = []
    names = []
    for eidx, (name, idx) in enumerate(indices.items()):
        color = colors[eidx]
        a_color = color_to_calpha(color, 0.)
        color = color_to_calpha(color, 1)
        scale.append([[0, a_color], [1, color]])
        name = f"{groups}: {name}"
        mean_values = np.mean(subdata[idx, ], axis=0)
        ys.append([name for _ in range(len(mean_values))])
        xs.append([str(p) for p in list(range(offset+1, subdata.shape[1] + offset+1))])
        names.append(name)
        means.append(mean_values)

    m_val = max([np.max(a) for a in means])
    for idx, (x, y, z) in enumerate(zip(xs, ys, means)):

        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                colorscale=scale[idx],
                name = names[idx],
                hovertemplate='<b>Fraction: %{x}</b><br><b>Protein Counts: %{z:.2e}</b> ',

            ),
            row=idx+1, col=1
        )
    fig.data[0].update(zmin=0, zmax=m_val)
    fig.data[1].update(zmin=0, zmax=m_val)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_traces(showscale=False)

    return fig


if __name__ == '__main__':
    from RDPMSpecIdentifier.datastructures import RDPMSpecData
    df = pd.read_csv("../testData/testFile.tsv", sep="\t", index_col=0)

    design = pd.read_csv("../testData/testDesign.tsv", sep="\t")
    df.index = df.index.astype(str)
    rdpmspec = RDPMSpecData(df, design, logbase=2)
    rdpmspec.normalize_array_with_kernel(kernel_size=3)
    array = rdpmspec.norm_array
    design = rdpmspec.internal_design_matrix

    fig = plot_replicate_distribution(array[0], design, "RNAse")
    fig.show()



