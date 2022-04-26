import pandas as pd
import numpy as np
import scipy.stats
import scipy.integrate


def chrome_ode(t, y, p):
    (
        mu_g_max,
        mu_d_max,
        K_g_Glc,
        K_I_Lac,
        K_d_Lac,
        k_Glc,
        k_Lac,
        k_Prod,
        feed_start,
        feed_end,
        Glc_feed_rate,
    ) = p

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


def predict_chrom_phase(model_param, process_param):
    feed_start, feed_end, Glc_feed_rate, Glc_0, VCD_0 = process_param
    mu_g_max, mu_d_max, K_g_Glc, K_I_Lac, K_d_Lac, k_Glc, k_Lac, k_Prod = model_param

    y0 = [VCD_0, Glc_0, 0, 0]
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
        Glc_feed_rate /24,
    )

    # integrates equations
    sol = scipy.integrate.solve_ivp(
        chrome_ode,
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


def generate_doe(num_exp: int, var_lims: dict, num_center_points=1, seed=123):
    rng = np.random.default_rng(seed)
    num_vars = len(var_lims)
    num_center_points = 1
    num_samples = num_exp - num_center_points
    # determine which vars are part of DOE
    doe_var = [1 if type(v) is list and v[0] < v[1] else 0 for v in var_lims.values()]
    doe_var_idx = np.cumsum(doe_var) - 1

    # sample points in the latin hypercube
    lhsampler = scipy.stats.qmc.LatinHypercube(d=sum(doe_var), centered=True, seed=rng)
    doe_plan = lhsampler.random(n=num_samples)

    # fill remaining unscaled vars
    doe_unscaled = np.ones([num_exp, num_vars]) * 0.5
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
            doe_scaled[:,i]=var_lims[k]

    return doe_scaled, doe_unscaled


def generate_data(var_lims, num_runs, filename="generated_owu.csv"):

    num_center_points = 1
    [doe_scaled, doe_unscaled] =  generate_doe(num_runs, var_lims, num_center_points)
    model_param_combinations = doe_scaled
    doe_design = pd.DataFrame(
        model_param_combinations, columns=[k for k in var_lims.keys()]
    )
    doe_normalized = pd.DataFrame(
        doe_unscaled, columns=[k for k in var_lims.keys()]
    )


    col_names = ["timesteps", "X:VCD", "X:Glc", "X:Lac", "X:Titer","W:Feed"]
    owu_df = pd.DataFrame(columns=col_names)

    i = 0
    for i in range(num_runs):
        mu_g_max = model_param_combinations[i, 0]
        mu_d_max = model_param_combinations[i, 1]
        K_g_Glc = model_param_combinations[i, 2]
        K_I_Lac = model_param_combinations[i, 3]
        K_d_Lac = model_param_combinations[i, 4]
        k_Glc = model_param_combinations[i, 5]
        k_Lac = model_param_combinations[i, 6]
        k_Prod = model_param_combinations[i, 7]
        feed_start = model_param_combinations[i, 8]
        feed_end = model_param_combinations[i, 9]
        Glc_feed_rate = model_param_combinations[i, 10]
        Glc_0 = model_param_combinations[i, 11]
        VCD_0 = model_param_combinations[i, 12]

        model_param = (
            mu_g_max,
            mu_d_max,
            K_g_Glc,
            K_I_Lac,
            K_d_Lac,
            k_Glc,
            k_Lac,
            k_Prod,
        )
        process_param = (feed_start, feed_end, Glc_feed_rate, Glc_0, VCD_0)

        t, y = predict_chrom_phase(model_param, process_param)
        
        time = np.array([t]).T /24
        xvar = y
        wvar = np.zeros_like(time)
        wvar[int(feed_start):int(feed_end),:] = Glc_feed_rate
        res =  np.hstack([time, xvar,wvar])
        
        owu_df = pd.concat(
            [owu_df, pd.DataFrame(res, columns=col_names)], ignore_index=True
        )


    owu_df.to_csv(filename, index=False)
    
    doe_design.to_csv(filename.replace(".csv","doe.csv"),index=False)
    doe_normalized.to_csv(filename.replace(".csv","doe_normalized.csv"),index=False)
    

    return owu_df
