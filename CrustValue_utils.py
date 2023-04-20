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

#from IPython import embed

# Sampling function
def distri_sample(distri_typ, P_abs_low, P_abs_high, nb_pts):

    P001 = P_abs_low # P001==0.1%
    P999 = P_abs_high # P999==99.9%
    if distri_typ == 'normal':
        mu = np.mean([P001, P999])
        sigma = (P999 - mu) / scipy.stats.norm.ppf(0.999)
        Plower = scipy.stats.norm.ppf(0.001, loc=mu, scale=sigma)
        Pupper = scipy.stats.norm.ppf(0.999, loc=mu, scale=sigma)
        nb_pts_tmp = nb_pts
        samples = []
        while len(samples)<nb_pts:
            samples_tmp = scipy.stats.norm.rvs(loc=mu, scale=sigma, size=int(nb_pts_tmp))
            idx = np.where((samples_tmp>Plower) & (samples_tmp<Pupper))
            samples = np.hstack((samples,samples_tmp[idx]))
            nb_pts_tmp = nb_pts-len(samples)

    elif distri_typ == 'lognormal':
        mu = np.mean(np.log([P001, P999]))
        s_sigma = (np.log(P999) - mu) / scipy.stats.norm.ppf(0.999) 
        s_mu = np.exp(mu) 
        Plower = scipy.stats.lognorm.ppf(0.001, s=s_sigma, loc=0.0, scale=s_mu)
        Pupper = scipy.stats.lognorm.ppf(0.999, s=s_sigma, loc=0.0, scale=s_mu)
        nb_pts_tmp = nb_pts
        samples = []
        while len(samples)<nb_pts:
            samples_tmp = scipy.stats.lognorm.rvs(s=s_sigma, loc=0.0, scale=s_mu, size=int(nb_pts_tmp))
            idx = np.where((samples_tmp>Plower) & (samples_tmp<Pupper))
            samples = np.hstack((samples,samples_tmp[idx]))
            nb_pts_tmp = nb_pts-len(samples)

    elif distri_typ == 'uniform':
        scale = P_abs_high-P_abs_low
        loc = P_abs_low
        samples = scipy.stats.uniform.rvs(loc=loc, scale=scale, size=int(nb_pts))

    return samples


def low_high_P10_P90_normal(P_low, P_high):
    P10 = P_low # P10==10.0%
    P90 = P_high # P90==90.0%
    mu = np.mean([P10, P90])
    sigma = (P90 - mu) / scipy.stats.norm.ppf(0.9)
    Plower = scipy.stats.norm.ppf(0.001, loc=mu, scale=sigma)
    Pupper = scipy.stats.norm.ppf(0.999, loc=mu, scale=sigma)
    return Plower, Pupper 

def low_high_P20_P80_normal(P_low, P_high):
    P20 = P_low # P20==20.0%
    P80 = P_high # P80==80.0%
    mu = np.mean([P20, P80])
    sigma = (P80 - mu) / scipy.stats.norm.ppf(0.8)
    Plower = scipy.stats.norm.ppf(0.001, loc=mu, scale=sigma)
    Pupper = scipy.stats.norm.ppf(0.999, loc=mu, scale=sigma)
    return Plower, Pupper 


# Area estimation from angle of repose
def distri_area_repose(distri_slope, cum_area_data):
    #area_max_angle = 0# np.interp(max_angle, cum_area_data[cum_area_data.files[0]], cum_area_data[cum_area_data.files[1]])
    distri_area = np.interp(distri_slope, cum_area_data[cum_area_data.files[0]], cum_area_data[cum_area_data.files[1]])
    #return distri_area-area_max_angle
    return distri_area


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


def distri_sum_elements(dict_vol, dict_conc, dict_price):
    value = 1
    value_mean_inputs = 1
    # Factor_1 for volume related inputs (area*thickness*density)
    factor_1 = 1
    factor_1_mult_means = 1
    for arg in dict_vol.values():
        factor_1 *= arg
        factor_1_mult_means *= np.array(arg).mean()
    # factor_2 for concentration and price pairs (c1*p1 + c2*p2 + ...)
    factor_2 = 0
    factor_2_sum_means = 0
    factor_2_all_means = np.zeros(len(dict_conc.values()))
    for i, ic in enumerate(dict_conc.values()):
        ip = list(dict_price.values())[i]
        factor_2 += (ic*ip)
        fac_mean = (np.array(ic).mean()*np.array(ip).mean())
        factor_2_sum_means += fac_mean
        factor_2_all_means[i] = fac_mean
    value = factor_1 * factor_2

    # Contributions for Tornado plot
    dict_ = dict_vol.copy() # collect all the input dictionaries in one
    dict_.update(dict_conc) 
    dict_.update(dict_price) 
    value_mean_inputs = factor_1_mult_means * factor_2_sum_means
    contri = {}
    # Factor_1 contributions
    for ik, key in enumerate(dict_vol.keys()):
        arg = list(dict_.values())[ik]
        if np.array(arg).size>1: # distribution array 
            P05 = distr_Px(arg, 0.95)
            P95 = distr_Px(arg, 0.05)
            min_ = (factor_1_mult_means / np.array(arg).mean() * P95) * factor_2_sum_means
            max_ = (factor_1_mult_means / np.array(arg).mean() * P05) * factor_2_sum_means
        elif np.array(arg).size==1: # single value
            min_ = value_mean_inputs
            max_ = value_mean_inputs
        contri[key] = {'percent': [-(value_mean_inputs-min_)/value_mean_inputs*100 , (max_-value_mean_inputs)/value_mean_inputs*100] }
    # Factor_2 concentration contributions 
    for ik, key in enumerate(dict_conc.keys()):
        arg = list(dict_conc.values())[ik]
        if np.array(arg).size>1: # distribution array 
            P05 = distr_Px(arg, 0.95)
            P95 = distr_Px(arg, 0.05)
            fixsum = np.delete(factor_2_all_means, ik).sum()
            ipm = np.array(list(dict_price.values())[ik]).mean()
            min_ = factor_1_mult_means * ( fixsum + (ipm*P95) )
            max_ = factor_1_mult_means * ( fixsum + (ipm*P05) )
        elif np.array(arg).size==1: # single value
            min_ = value_mean_inputs
            max_ = value_mean_inputs
        contri[key] = {'percent': [-(value_mean_inputs-min_)/value_mean_inputs*100 , (max_-value_mean_inputs)/value_mean_inputs*100] }
    # Factor_2 price contributions 
    for ik, key in enumerate(dict_price.keys()):
        arg = list(dict_price.values())[ik]
        if np.array(arg).size>1: # distribution array 
            P05 = distr_Px(arg, 0.95)
            P95 = distr_Px(arg, 0.05)
            fixsum = np.delete(factor_2_all_means, ik).sum()
            icm = np.array(list(dict_conc.values())[ik]).mean()
            min_ = factor_1_mult_means * ( fixsum + (icm*P95) )
            max_ = factor_1_mult_means * ( fixsum + (icm*P05) )
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

def distri_plot(samples, plot_type='pdf', xlabel='x', nb_bar=100, html_fig=False):

    # Plot - Configuration
    #nb_bar = 100
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

    if html_fig is not False:
        fig.write_html(html_fig)



def distri_cum_multiplot(traces, leg_names, leg_title='',  xlabel='x', html_fig=False):
    ylabel = 'Cumulative probability' 
    fig = go.Figure()
    for i_, samples in enumerate(traces): 
        # Calculation of the cumulative distribution (simple maths) 
        samples_sorted = np.sort(samples)
        samples_cum = 1. * np.arange(len(samples_sorted)) / (len(samples_sorted)-1)
        # Estimation of the quantiles (percentiles) P10 P50 P90
        y_Pxx = [0.1, 0.5, 0.9]
        x_Pxx = np.interp(y_Pxx, samples_cum, samples_sorted) # x and y needs to be flipped for np.interp
        x_Pxx = x_Pxx[::-1]
        anno_Pxx = ['<b> P10 (', '<b> P50 (', '<b> P90 (']
        # Plot the result
        fig.add_trace(go.Scatter(x=samples_sorted, y=1-samples_cum, name=leg_names[i_]))
        fig.update_layout(showlegend=True)
        #fig.add_trace(go.Scatter(x=np.hstack((samples_sorted,samples_sorted[0])) , y=np.hstack((1-samples_cum, 0)), fill="toself", mode= 'none'))
    # Reverse trace for coloring (largest x-values curve is blue and then red...)
    fig.data=fig.data[::-1]
    # Add Pxx points and texts
    anno_text = [anno_Pxx[i]+str(np.format_float_scientific(k, precision=2, exp_digits=1))+')</b>' for i,k in enumerate(x_Pxx)]
    fig.add_trace(go.Scatter(x=x_Pxx, y=y_Pxx, mode="markers+text", line_color='black', text=anno_text, textposition="middle right", showlegend=False))    
    # Add Pxx lines for style
    #for j in range(len(x_Pxx)):
    #    fig.add_trace(go.Scatter(x=[samples.min(), x_Pxx[j]], y=[y_Pxx[j], y_Pxx[j]] , mode="lines", line_color='black',showlegend=False))
    fig.update_layout(xaxis_title="<b>"+xlabel+"</b>", 
                    yaxis_title="<b>"+ylabel+"</b>",
                    legend_title=leg_title)
    fig.update_xaxes(domain=(0.25, 0.75))
    fig.show()
    if html_fig is not False:
        fig.write_html(html_fig)


def contri_plot(contri, xlabel='x', max_first=0, html_fig=False):
     
    max_ = [(np.abs(all_val['percent'])).max() for all_val in list(contri.values())]
    order = np.argsort(max_)
    if max_first>0:
        order = order[-max_first::]
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
    fig.update_layout(xaxis_title="<b>"+xlabel+"</b>", barmode='relative', showlegend=False,
                      autosize=True)
    fig.show()

    if html_fig is not False:
        fig.write_html(html_fig)

    return



def plot_sunburst(Ws, Cs, Ps, elements, lists,labels,Px,title,html_fig):

    # Create empty indx_ and list_ to map the elements lists to plot   
    indx_  = {}
    list_ = {}
    for i in range(len(lists)):
        indx_[str(i)] = []
        list_[str(i)] = []

    # fill up indx_ and list_
    for ie, e in enumerate(elements):
        for ilist in range(len(lists)):
            if np.any(np.isin(lists[ilist],e)):
                indx_[str(ilist)].append(int(ie))
                list_[str(ilist)].append(e)


    for il in range(len(lists)):
        indx_easy =[]
        indx_rees =[]
        indx_others = []
        list_easy = []
        list_rees = []
        list_others = []

    for ie, e in enumerate(elements):
        if np.any(np.isin(easy_ref,e)):
            indx_easy.append(int(ie))
            list_easy.append(e)
        elif np.any(np.isin(rees,e)):
            indx_rees.append(int(ie))
            list_rees.append(e)
        else:
            indx_others.append(int(ie))
            list_others.append(e)

    return

