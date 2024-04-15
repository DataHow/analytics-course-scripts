from google.colab import output
from ipywidgets import interact, interactive,fixed
import ipywidgets as widgets
from sklearn import datasets
from pyDOE2 import fullfact, ff2n, ccdesign
import definitive_screening_design as dsd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy.integrate
import scipy.stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.templates.default = "plotly_white"
output.enable_custom_widget_manager()


FEED_START= widgets.FloatRangeSlider(
    value=[1, 4],
    min=0,
    max=6,
    step=1,
    description='feed_start',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='',
)
FEED_END = widgets.FloatRangeSlider(
    value=[8, 12],
    min=6,
    max=14,
    step=1,
    description='feed_end',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='',
)
FEED_RATE = widgets.FloatRangeSlider(
    value=[5, 20],
    min=1,
    max=30,
    step=0.5,
    description='feed_rate',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='',
)
GLC_0 = widgets.FloatRangeSlider(
    value=[10, 40],
    min=1,
    max=80,
    step=0.5,
    description='glc_0',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
VCD_0 = widgets.FloatRangeSlider(
    value=[0.1, 1.0],
    min=0.1,
    max=2.0,
    step=0.1,
    description='vcd_0',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
NUM_RUNS = widgets.IntSlider(
    value=40,
    min=1,
    max=100,
    step=1,
    description='num_runs',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)
DOE_DESIGN = widgets.Select(
    options=['Latin-Hypercube Sampling', '2-level Full-Factorial', '3-level Full-Factorial','Central-Composite','Definitive-Screening'],
    value='Latin-Hypercube Sampling',
    rows=5,
    description='doe_design',
    disabled=False
)



SELECT_RUNS = widgets.SelectMultiple(
    options=list(range(100)),
    value=[0,1],
    rows=10,
    description='select_runs',
    disabled=False
)


HIGHLIGHT_RUN = widgets.IntSlider(
    value=0,
    min=0,
    max=99,
    step=1,
    description='highlight run',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

SELECT_COLOR = widgets.Select(
    options=["run id","titer_14","feed_start","feed_end","feed_rate","glc_0","vcd_0"],
    value="titer_14",
    rows=7,
    description='select color',
    disabled=False
)


def process_ode(t, y, p):
    (mu_g_max, mu_d_max, K_g_Glc, K_I_Lac, K_d_Lac, k_Glc, k_Lac, k_Prod, feed_start, feed_end, Glc_feed_rate) = p

    VCD, Glc, Lac, titer = y[0], y[1], y[2], y[3]
    MM_Glc = Glc / (K_g_Glc + Glc)
    mu_g = mu_g_max * MM_Glc * K_I_Lac / (K_I_Lac + Lac)
    phi = np.exp(0.1 * (Glc - 75.0))
    mu_d = mu_d_max * (1.0 + phi / (1.0 + phi)) * Lac / (K_d_Lac + Lac)
    growth_ratio = mu_g / mu_g_max

    # compute mass balances
    Glc_Min = Glc / (0.05 + Glc)
    dVCDdt = (mu_g - mu_d) * VCD
    dGlcdt = -k_Glc * Glc_Min * VCD
    dLacdt = k_Lac * VCD
    dTiterdt = k_Prod * MM_Glc * ((1.0 - growth_ratio) ** 2.0) * VCD

    # add feed rate
    if feed_end >= t >= feed_start:
        dGlcdt += Glc_feed_rate

    return [dVCDdt, dGlcdt, dLacdt, dTiterdt]


def predict_process(feed_start, feed_end, feed_rate, glc_0, vcd_0):

    MU_G_MAX = 0.05
    MU_D_MAX = 0.025
    K_G_GLC  = 1
    K_I_LAC  = 30
    K_D_LAC  = 50
    K_GLC    = 0.04
    K_LAC    = 0.06
    K_PROD   = 1
    MODEL_PARAM = [MU_G_MAX,MU_D_MAX,K_G_GLC,K_I_LAC,K_D_LAC,K_GLC,K_LAC,K_PROD]

    mu_g_max, mu_d_max, K_g_Glc, K_I_Lac, K_d_Lac, k_Glc, k_Lac, k_Prod = MODEL_PARAM


    y0 = [vcd_0, glc_0, 0, 0]
    t_start, t_end = 0, 24 * 14
    t_span = np.arange(t_start, t_end + 24, 24)
    p = (
        mu_g_max,
        mu_d_max,
        K_g_Glc,
        K_I_Lac,
        K_d_Lac,
        k_Glc,
        k_Lac,
        k_Prod,
        24.0 * feed_start,
        24.0 * feed_end,
        feed_rate /24,
    )

    # integrates equations
    sol = scipy.integrate.solve_ivp(
        process_ode,
        t_span=[t_start, t_end],
        y0=y0,
        t_eval=t_span,
        method="BDF",
        args=([p]),
        rtol=1e-6,
        atol=1e-6,
    )

    t = sol.t.tolist()
    y = sol.y.T.tolist()

    return t, y

def plot_process(feed_start, feed_end, feed_rate, glc_0, vcd_0):
    t,x = predict_process(feed_start, feed_end, feed_rate, glc_0, vcd_0)

    time = np.array([t]).T /24
    xvar = x
    wvar = np.zeros_like(time)
    wvar[int(feed_start):int(feed_end),:] = feed_rate
    res =  np.hstack([time, xvar,wvar])

    run = pd.DataFrame(res, columns=["Time","X:VCD", "X:Glc", "X:Lac", "X:Titer","W:Feed"])

    fig = make_subplots(rows=1, cols=5, subplot_titles=run.columns[1:])
    for i,column in enumerate(run.columns):
        if i==0:
            continue
        fig.add_trace(go.Scatter(x=run["Time"].values, y=run[column].values),row=1,col=i)
    fig.update_layout(showlegend=False, title_text="Process variables evolution for generated run",width=1600)
    fig.show()


def generate_doe(feed_start, feed_end, feed_rate, glc_0, vcd_0, num_runs, doe_design="Latin-Hypercube Sampling", num_center_points=1, seed=123):
    var_lims = {
        "feed_start": feed_start,
        "feed_end": feed_end,
        "fee_rate": feed_rate,
        "glc_0": glc_0,
        "vcd_0": vcd_0
    }

    rng = np.random.default_rng(seed)
    num_vars = len(var_lims)
    num_center_points = 1
    if type(num_runs) is tuple:
        num_runs = num_runs[0]
    num_samples = num_runs - num_center_points
    # determine which vars are part of DOE
    doe_var = [1 if type(v) is tuple and v[0] < v[1] else 0 for v in var_lims.values()]
    doe_var_idx = np.cumsum(doe_var) - 1

    if doe_design == "Latin-Hypercube Sampling":
        # sample points in the latin hypercube
        lhsampler = scipy.stats.qmc.LatinHypercube(d=sum(doe_var), scramble=False, seed=rng)
        doe_plan = lhsampler.random(n=num_samples)
    if doe_design == "2-level Full-Factorial":
        doe_plan = (ff2n(sum(doe_var))+1)/2
    if doe_design == "3-level Full-Factorial":
        doe_plan = fullfact([3]*sum(doe_var))/2
    if doe_design == "Central-Composite":
        doe_plan = (ccdesign(sum(doe_var),center=(0,1), face='cci')+1)/2
    if doe_design == "Definitive-Screening":
        doe_plan = (dsd.generate(n_num=sum(doe_var),verbose=False).values+1)/2
    if num_runs < len(doe_plan):
        print(f"\n The selected design requires {len(doe_plan)} euns, while only {num_runs} runs were selected!")
        print(" Only a subset of created runs will be used!\n")
        subset_idx = np.sort(np.random.choice(range(len(doe_plan)),size=num_runs, replace=False))
        doe_plan= doe_plan[subset_idx,:]


    # fill remaining unscaled vars
    doe_unscaled = np.ones([len(doe_plan)+1, num_vars]) * 0.5
    for i in range(num_vars):
        if doe_var[i] == 1:
            doe_unscaled[num_center_points:, i] = doe_plan[:, doe_var_idx[i]]


    # scale all vars according to var_lims
    doe_scaled = doe_unscaled
    for i, k in enumerate(var_lims.keys()):
        if doe_var[i] == 1:
            doe_scaled[:, i] = (
                doe_unscaled[:, i] * (var_lims[k][1] - var_lims[k][0]) + var_lims[k][0]
            )
        else:
            doe_scaled[:,i]=var_lims[k][0]

    return doe_scaled


def plot_doe(feed_start, feed_end, feed_rate, glc_0, vcd_0, num_runs, doe_design):

    doe_scaled = generate_doe(feed_start=feed_start, feed_end=feed_end, feed_rate=feed_rate, glc_0=glc_0, vcd_0=vcd_0, num_runs=num_runs, doe_design=doe_design)

    # generate_data(doe_scaled, feed_start, feed_end, feed_rate, glc_0, vcd_0, num_runs, doe_design)

    fig = go.Figure(data=go.Parcoords(
        line_color='blue',
        dimensions = list([
                dict(label = 'feed_start',values = doe_scaled[:,0].round(),range=list(feed_start)),
                dict(label = 'feed_end', values = doe_scaled[:,1].round(),range=list(feed_end)),
                dict(label = 'feed_rate',values = doe_scaled[:,2],range=list(feed_rate)),
                dict(label = 'glc_0',values = doe_scaled[:,3],range=list(glc_0)),
                dict(label = 'vcd_0',values = doe_scaled[:,4],range=list(vcd_0)),
            ])
    ))
    fig.update_layout(showlegend=False, title_text=f"Generated design of experiments using {doe_design} with {len(doe_scaled)} runs",width=1600)

    fig.show()
    return doe_scaled


def generate_data(doe_scaled, filename="/content/generated_owu.csv"):

    col_names = ["timesteps", "X:VCD", "X:Glc", "X:Lac", "X:Titer","W:Feed"]
    doe_names = ["feed_start", "feed_end", "feed_rate", "glc_0","vcd_0"]

    doe_df = pd.DataFrame(doe_scaled, columns = doe_names)
    owu_df = pd.DataFrame(columns = col_names)

    num_runs = len(doe_scaled)

    i = 0
    for i in range(num_runs):
        feed_start = doe_scaled[i, 0].round()
        feed_end = doe_scaled[i, 1].round()
        feed_rate = doe_scaled[i, 2]
        glc_0 = doe_scaled[i, 3]
        vcd_0 = doe_scaled[i, 4]

        t, x = predict_process(feed_start, feed_end, feed_rate, glc_0, vcd_0)

        time = np.array([t]).T /24
        xvar = x
        wvar = np.zeros_like(time)
        wvar[int(feed_start):int(feed_end),:] = feed_rate
        res =  np.hstack([time, xvar,wvar])

        owu_df = pd.concat(
            [owu_df, pd.DataFrame(res, columns=col_names)], ignore_index=True
        )


    owu_df.to_csv(filename, index=False)

    doe_df.to_csv(filename.replace("_owu.csv","_doe.csv"),index=False)

    owu_df.index = pd.MultiIndex.from_product([list(range(num_runs)),list(range(15))], names=["run","time"])
    return owu_df, doe_df



def plot_data(owu_df, select_runs):
    fig = make_subplots(rows=1, cols=5, subplot_titles=owu_df.columns[1:])
    for j in select_runs:
        plot_run_ix = owu_df.index.get_level_values("run") == j
        for i,c in enumerate(owu_df.columns):
            if i==0:
                continue
            fig.add_trace(go.Scatter(x=list(range(15)), y=owu_df[c].values[plot_run_ix], name="Run = " + str(j),marker=dict(color=px.colors.qualitative.Plotly[j % 10])), row=1, col=i)
    fig.update_layout(showlegend=False, title_text="Process variable evolution for selected runs", width=1600)
    fig.show()

def plot_data_color(owu_df, doe_df, highlight_run=0, select_color="run id"):

    def get_color(colorscale_name, loc):
        from _plotly_utils.basevalidators import ColorscaleValidator

        # first parameter: Name of the property being validated
        # second parameter: a string, doesn't really matter in our use case
        cv = ColorscaleValidator("colorscale", "")
        # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
        colorscale = cv.validate_coerce(colorscale_name)

        if hasattr(loc, "__iter__"):
            return [get_continuous_color(colorscale, x) for x in loc]
        return get_continuous_color(colorscale, loc)


    def get_continuous_color(colorscale, intermed):
        import plotly.colors
        from PIL import ImageColor

        """
        Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
        color for any value in that range.

        Plotly doesn't make the colorscales directly accessible in a common format.
        Some are ready to use:

            colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

        Others are just swatches that need to be constructed into a colorscale:

            viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
            colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

        :param colorscale: A plotly continuous colorscale defined with RGB string colors.
        :param intermed: value in the range [0, 1]
        :return: color in rgb string format
        :rtype: str
        """
        if len(colorscale) < 1:
            raise ValueError("colorscale must have at least one color")

        hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

        if intermed <= 0 or len(colorscale) == 1:
            c = colorscale[0][1]
            return c if c[0] != "#" else hex_to_rgb(c)
        if intermed >= 1:
            c = colorscale[-1][1]
            return c if c[0] != "#" else hex_to_rgb(c)

        for cutoff, color in colorscale:
            if intermed > cutoff:
                low_cutoff, low_color = cutoff, color
            else:
                high_cutoff, high_color = cutoff, color
                break

        if (low_color[0] == "#") or (high_color[0] == "#"):
            # some color scale names (such as cividis) returns:
            # [[loc1, "hex1"], [loc2, "hex2"], ...]
            low_color = hex_to_rgb(low_color)
            high_color = hex_to_rgb(high_color)

        return plotly.colors.find_intermediate_color(
            lowcolor=low_color,
            highcolor=high_color,
            intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
            colortype="rgb",
        )



    if select_color == "run id":
        color_idx=np.repeat(np.array(list(range(len(doe_df)))),15)
    if select_color == "titer_14":
        color_idx =np.repeat(np.array(owu_df["X:Titer"][:,14]),15)
    if select_color in doe_df.columns:
        color_idx = np.repeat(np.array(doe_df[select_color]),15)
    owu_df["color"] = color_idx


    owu_columns = owu_df.columns[1:-1]
    fig = make_subplots(rows=1, cols=5, subplot_titles=owu_columns)
    # for i,c in enumerate(owu_columns):
    #     for j in range(len(doe_df)):
    #         plot_run_ix = owu_df.index.get_level_values("run") == j
    #         fig.add_trace(go.Scatter(x=list(range(15)), y=owu_df[c].values[plot_run_ix], name="Run = " + str(j),marker=dict(color=px.colors.qualitative.Plotly[j % 10])), row=1, col=i+1)

    for i,c in enumerate(owu_columns):
        fig.add_trace(go.Scatter(x=list(range(15)),y=owu_df[c],mode='markers',marker=dict(size=0,color="rgba(0,0,0,0)",colorscale='Portland',cmin=min(color_idx),cmax=max(color_idx),colorbar=dict(thickness=40, title=str(select_color))),showlegend=False), row=1, col=i+1)
        for color_val in np.unique(color_idx):
            color_val_norm = (color_val -min(color_idx)) / (max(color_idx)-min(color_idx))
            owu_subset = owu_df[owu_df['color']==color_val]
            # print(owu_subset)
            fig.add_trace(go.Scatter(x=list(range(15)),y=owu_subset[c],mode='lines+markers',name="Run id = " + str(owu_df.index.get_level_values(0)[color_val == color_idx][0]),marker=dict(color=get_color('Portland',color_val_norm))), row=1, col=i+1)



    highlight_run_ix = owu_df.index.get_level_values("run") == highlight_run
    if highlight_run > len(doe_df):
        print("\n Highlighted run is higher than number of performed experiments \n")
    for i,c in enumerate(owu_columns):
      fig.add_trace(go.Scatter(x=list(range(15)), y=owu_df[c].values[highlight_run_ix],name="Run = "+ str(highlight_run), marker=dict(color="black",size=10)),row=1, col=i+1)
    fig.update_layout(showlegend=False, title_text=f"Process variable evolution for selected runs and values colored by {select_color}", width=1600)
    fig.show()
