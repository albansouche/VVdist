# Imports
import numpy as np
import scipy.stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
if 'google.colab' in str(get_ipython()):  # used to configure plotly rendering in jupyter notebook or google colab
  pio.renderers.default='colab'
else:
  pio.renderers.default='notebook'

# Sampling function
def distri_sample(distri_typ, P90_low, P10_high, nb_pts):

    P10 = P90_low
    P90 = P10_high
    if distri_typ == 'normal':
        mu = np.mean([P10, P90])
        sigma = (P90 - mu) / scipy.stats.norm.ppf(0.90)
        Plower = scipy.stats.norm.ppf(0.0001, loc=mu, scale=sigma,)
        Pupper = scipy.stats.norm.ppf(0.9999, loc=mu, scale=sigma)
        nb_pts_tmp = nb_pts
        samples = []
        while len(samples)<nb_pts:
            samples_tmp = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=int(nb_pts_tmp))
            idx = np.where((samples_tmp>Plower) & (samples_tmp<Pupper))
            samples = np.hstack((samples,samples_tmp[idx]))
            nb_pts_tmp = nb_pts-len(samples)

    elif distri_typ == 'lognormal':
        mu = np.mean(np.log([P10, P90]))
        s_sigma = (np.log(P90) - mu) / scipy.stats.norm.ppf(0.90) 
        s_mu = np.exp(mu) 
        Plower = scipy.stats.lognorm.ppf(0.00001, s=s_sigma, loc=0.0, scale=s_mu)
        Pupper = scipy.stats.lognorm.ppf(0.999, s=s_sigma, loc=0.0, scale=s_mu)
        nb_pts_tmp = nb_pts
        samples = []
        while len(samples)<nb_pts:
            samples_tmp = scipy.stats.lognorm.rvs(s=s_sigma, loc=0.0, scale=s_mu, size=int(nb_pts_tmp))
            idx = np.where((samples_tmp>Plower) & (samples_tmp<Pupper))
            samples = np.hstack((samples,samples_tmp[idx]))
            nb_pts_tmp = nb_pts-len(samples)

    return samples

def distri_multiply(dict_):
    value = 1
    value_mean_inputs = 1
    for arg in dict_.values():
        value *= arg
        value_mean_inputs *= np.array(arg).mean()
    contri = {}
    for ik, key in enumerate(dict_.keys()):
        arg = list(dict_.values())[ik]
        if np.array(arg).size>1: # distribution array 
            P05 = distr_Px(arg, 0.95)
            P95 = distr_Px(arg, 0.05)
            min_ = value_mean_inputs / np.array(arg).mean() * P95
            max_ = value_mean_inputs / np.array(arg).mean() * P05    
        elif np.array(arg).size==1: # single value
            min_ = value_mean_inputs
            max_ = value_mean_inputs
        contri[key] = {'percent': [-(value_mean_inputs-min_)/value_mean_inputs*100 , (max_-value_mean_inputs)/value_mean_inputs*100] }
    return value, contri

def distri_P10_high(samples):
    return distr_Px(samples, 0.9)

def distri_P90_low(samples):
    return distr_Px(samples, 0.1)

def distr_Px(samples, pval):
    samples_sorted = np.sort(samples)
    samples_cum = 1. * np.arange(len(samples_sorted)) / (len(samples_sorted)-1)
    fval = np.interp(pval, samples_cum, samples_sorted) # x and y needs to be flipped for np.interp
    return fval

def distri_plot(samples, plot_type='pdf', xlabel='x'):

    # Plot - Configuration
    nb_bar = 100
    bin_size = (samples.max()-samples.min()) / nb_bar
    histo_norm = 'probability'  # '' 'probability' 'density' 'probability density' (used by plotly histogram)

    if plot_type=='pdf':

        ylabel    = 'Probability'  # y-axis label
        fig = go.Figure( go.Histogram(x=samples, xbins=dict(size=bin_size), histnorm=histo_norm) )
        fig.update_layout(xaxis_title="<b>"+xlabel+"</b>", 
                        yaxis_title="<b>"+ylabel+"</b>",
                        showlegend=False )
        fig.update_xaxes(domain=(0.25, 0.75))

    elif plot_type=='cdf':

        ylabel = 'Cumulative probability' 
        #fig = go.Figure(go.Histogram(x=samples, cumulative_enabled=True, xbins=dict(size=bin_size), histnorm=histo_norm))
        fig = go.Figure()
        # Calculation of the cumulative distribution (simple maths) 
        samples_sorted = np.sort(samples)
        samples_cum = 1. * np.arange(len(samples_sorted)) / (len(samples_sorted)-1)
        # Estimation of the quantiles (percentiles) P10 P50 P90
        y_Pxx = [0.1, 0.5, 0.9]
        x_Pxx = np.interp(y_Pxx, samples_cum, samples_sorted) # x and y needs to be flipped for np.interp
        x_Pxx = x_Pxx[::-1]
        anno_Pxx = ['<b> P10 (', '<b> P50 (', '<b> P90 (']
        # Plot the results
        fig.add_trace(go.Scatter(x=samples_sorted, y=1-samples_cum))
        fig.add_trace(go.Scatter(x=np.hstack((samples_sorted,samples_sorted[0])) , y=np.hstack((1-samples_cum, 0)), fill="toself", mode= 'none'))
        # Add Pxx points and texts
        anno_text = [anno_Pxx[i]+str(np.format_float_scientific(k, precision=2, exp_digits=1))+')</b>' for i,k in enumerate(x_Pxx)]
        fig.add_trace(go.Scatter(x=x_Pxx, y=y_Pxx, mode="markers+text", line_color='black', text=anno_text, textposition="middle right"))
        # Add Pxx lines for style
        for j in range(len(x_Pxx)):
            fig.add_trace(go.Scatter(x=[samples.min(), x_Pxx[j]], y=[y_Pxx[j], y_Pxx[j]] , mode="lines", line_color='black'))
        fig.update_layout(xaxis_title="<b>"+xlabel+"</b>", 
                        yaxis_title="<b>"+ylabel+"</b>",
                        showlegend=False)
        fig.update_xaxes(domain=(0.25, 0.75))
    fig.show()

def contri_plot(contri, xlabel='x'):

     
    max_ = [(np.abs(all_val['percent'])).max() for all_val in list(contri.values())]
    order = np.argsort(max_)
    fig = go.Figure()

    for i_ in order:
    #for key_, vals_ in contri.items():
        key_ = list(contri.keys())[i_]
        vals_= list(contri.values())[i_]
        
        fig.add_trace(go.Bar(
                x=[vals_['percent'][0]],
                y=["<b>"+key_+"</b>"],
                marker_color='rgb(255,132,122)',
                width=0.4,
                orientation='h'))
        fig.add_trace(go.Bar(
                x=[vals_['percent'][1]],
                y=["<b>"+key_+"</b>"],
                marker_color='rgb(123, 255, 153)',
                width=0.4,
                orientation='h'))
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1.5, opacity=0.6)
    fig.update_layout(xaxis_title="<b>"+xlabel+"</b>", barmode='relative', showlegend=False)
    fig.show()

    return

