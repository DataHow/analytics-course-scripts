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


def plot_ensemble(X_train, y_train, X_test, y_test, y_hat, model, upper, lower):
    """
    Function to help plotting the prediction intervals of the ensemble model.
    """
    import numpy as np
    import plotly.graph_objects as go
    import statistics
    from scipy.stats import norm

    y_pred = model.predict(X_train.reshape(-1, 1))
    upper = y_pred + upper
    lower = y_pred + lower
    residuals = y_train - y_pred
    param_upper = (
        y_pred
        + np.mean(residuals)
        + 1 / norm.cdf(0.95) * np.sqrt(statistics.variance(residuals))
    )
    param_lower = (
        y_pred
        + np.mean(residuals)
        - 1 / norm.cdf(0.95) * np.sqrt(statistics.variance(residuals))
    )
    fig = go.Figure(
        [
            go.Scatter(name="Data", x=X_test, y=y_test, mode="markers"),
            go.Scatter(
                name="Fit",
                x=X_test,
                y=model.predict(X_test.reshape(-1, 1)),
                mode="lines",
                line=dict(color="rgb(31, 119, 180)"),
            ),
            go.Scatter(
                name="Upper Bound",
                x=X_train,
                y=upper,
                mode="lines",
                marker=dict(color="rgba(68, 68, 68, 0.3)"),
                line=dict(width=0),
                showlegend=False,
            ),
            go.Scatter(
                name="Bootstrap PI",
                x=X_train,
                y=lower,
                marker=dict(color="rgba(68, 68, 68, 0.3)"),
                line=dict(width=0),
                mode="lines",
                fillcolor="rgba(68, 68, 68, 0.3)",
                fill="tonexty",
            ),
            go.Scatter(
                name="Parametric Upper Bound",
                x=X_train,
                y=param_upper,
                mode="lines",
                marker=dict(color="rgba(255, 99, 71, 0.5)"),
                line=dict(width=1),
                showlegend=False,
            ),
            go.Scatter(
                name="Parametric CI",
                x=X_train,
                y=param_lower,
                marker=dict(color="rgba(255, 99, 71, 0.5)"),
                line=dict(width=1),
                mode="lines",
                fillcolor="rgba(255, 99, 71, 0.5)",
                fill="tonexty",
            ),
        ]
    )
    fig.update_layout(hovermode="x")
    fig.show()
