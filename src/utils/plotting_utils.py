from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import cycle
from plotly.colors import n_colors
import pandas as pd
from statsmodels.tsa.stattools import pacf, acf
import plotly.graph_objects as go
import warnings
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff


def make_lines_greyscale(fig):
    colors = cycle(list(set(n_colors('rgb(100, 100, 100)', 'rgb(200, 200, 200)', 2+1, colortype='rgb'))))
    for d in fig.data:
        d.line.color = next(colors)
    return fig

def two_line_plot_secondary_axis(x, y1, y2, y1_name="y1", y2_name="y2", title="", legends = None, xlabel="Time", ylabel="Value", greyscale=False, dash_secondary=False):
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=x, y=y1, name=y1_name),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=x, y=y2, name=y2_name, line = dict(dash='dash') if dash_secondary else None),
        secondary_y=True,
    )
    if legends:
        names = cycle(legends)
        fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_layout(
            autosize=False,
            width=900,
            height=500,
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            title_text=title,
            titlefont={
                "size": 20
            },
            legend_title = None,
            yaxis=dict(
                title_text=ylabel,
                titlefont=dict(size=12),
            ),
            xaxis=dict(
                title_text=xlabel,
                titlefont=dict(size=12),
            )
        )
    if greyscale:
        fig = make_lines_greyscale(fig)
    return fig

def multiple_line_plot_secondary_axis(df, x, primary, secondary, color_or_linetype, title="", use_linetype=False, greyscale=False):
    df = pd.pivot_table(df, index=x, columns=color_or_linetype, values=[primary, secondary]).reset_index()
    df.columns = [str(c1)+"_"+str(c2) if c2!="" else c1 for c1, c2 in df.columns]
    primary_columns = sorted([c for c in df.columns if primary in c])
    secondary_columns = sorted([c for c in df.columns if secondary in c])
    if use_linetype:
        colors = ["solid","dash","dot","dashdot"]
    else:
        colors = px.colors.qualitative.Plotly
        if len(primary_columns)>=len(colors):
            colors = px.colors.qualitative.Light24
    assert len(primary_columns)<=len(colors)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    for c, color in zip(primary_columns, colors[:len(primary_columns)]):
        fig.add_trace(
            go.Scatter(x=df[x], y=df[c], name=c, line=dict(dash=color) if use_linetype else dict(color=color)),
            secondary_y=False,
        )
    for c, color in zip(secondary_columns, colors[:len(primary_columns)]):
        fig.add_trace(
            go.Scatter(x=df[x], y=df[c], name=c, line=dict(dash=color) if use_linetype else dict(color=color)),
            secondary_y=True,
        )
    # Add figure title
    fig.update_layout(
        title_text=title
    )
    if greyscale:
#         colors = cycle(list(set(px.colors.sequential.Greys[1:])))
        colors = cycle(list(set(n_colors('rgb(100, 100, 100)', 'rgb(200, 200, 200)', 2+1, colortype='rgb'))))
        for d in fig.data:
            d.line.color = next(colors)
    return fig

def hex_to_rgb(hex):
    if hex.startswith("#"):
        hex = hex.lstrip("#")
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def plot_autocorrelation(series,vertical=False, figsize=(500, 900), **kwargs):
    if "qstat" in kwargs.keys():
        warnings.warn("`qstat` for acf is ignored as it has no impact on the plots")
        kwargs.pop("qstat")
    acf_args = ["adjusted","nlags", "fft", "alpha", "missing"]
    pacf_args = ["nlags","method","alpha"]
    if "nlags" not in kwargs.keys():
        nobs = len(series)
        kwargs['nlags'] = min(int(10 * np.log10(nobs)), nobs // 2 - 1)
    kwargs['fft'] = True
    acf_kwargs = {k:v for k,v in kwargs.items() if k in acf_args}
    pacf_kwargs = {k:v for k,v in kwargs.items() if k in pacf_args}
    acf_array = acf(series, **acf_kwargs)
    pacf_array = pacf(series, **pacf_kwargs)
    is_interval = False
    if "alpha" in kwargs.keys():
        acf_array, _ = acf_array
        pacf_array, _ = pacf_array
    x_ = np.arange(1,len(acf_array))
    rows, columns = (2, 1) if vertical else (1,2)
    fig = make_subplots(
            rows=rows, cols=columns, shared_xaxes=True, shared_yaxes=False, subplot_titles=['Autocorrelation (ACF)', 'Partial Autocorrelation (PACF)']
        )
    #ACF
    row, column = 1, 1
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,acf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(acf_array))]
    fig.append_trace(go.Scatter(x=x_, y=acf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    #PACF
    row, column = (2,1) if vertical else (1,2)
    [fig.append_trace(go.Scatter(x=(x,x), y=(0,pacf_array[x]), mode='lines',line_color='#3f3f3f'), row=row, col=column) 
     for x in range(1, len(pacf_array))]
    fig.append_trace(go.Scatter(x=x_, y=pacf_array[1:], mode='markers', marker_color='#1f77b4',
                   marker_size=8), row=row, col=column)
    fig.update_traces(showlegend=False)
    fig.update_yaxes(zerolinecolor='#000000')
    fig.update_layout(
            autosize=False,
            width=figsize[1],
            height=figsize[0],
            title={
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            titlefont={
                "size": 20
            },
            legend_title = None,
            yaxis=dict(
                titlefont=dict(size=12),
            ),
            xaxis=dict(
                titlefont=dict(size=12),
            )
        )
    return fig

def show_plotly_swatches():
    fig = px.colors.qualitative.swatches()
    fig.show()

def plot_correlation_plot(df, title="Heatmap", num_decimals=2, figsize=(200,200)):
    df = df.round(num_decimals)
    mask = np.triu(np.ones_like(df, dtype=bool))
    df_mask = df.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                                      x=df_mask.columns.tolist(),
                                      y=df_mask.columns.tolist(),
                                      colorscale=px.colors.diverging.RdBu,
                                      hoverinfo="none", #Shows hoverinfo for null values
                                      showscale=True, ygap=1, xgap=1
                                     )
    fig.update_xaxes(side="bottom")
    fig.update_layout(
        title_text=title, 
        title_x=0.5, 
        width=figsize[0], 
        height=figsize[1],
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    return fig