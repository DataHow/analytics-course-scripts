import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm
from bayes_opt import BayesianOptimization, UtilityFunction

from functools import partial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from matplotlib import gridspec

def polynomial_function(x,eps=0):
  x = np.array(x)
  return -1*np.power(x,4)+0.5*np.power(x,3)+6*np.power(x,2)+0.5*np.power(x,1)-6+eps*np.random.normal(0,1,x.shape)

def exp_function(x,eps=0):
  x = np.array(x)
  return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def sin_function(x,eps=0):
  x = np.array(x)
  return np.sin(x)/2 - ((10 - x)**2)/50 + 2

def define_originator_function(choice):
  if choice == "polynomial":
    fun = polynomial_function
    domain = (-3,3)
    counterdomain = (-20,20)
  elif choice == "sin":
    fun = sin_function
    domain = (0,20)
    counterdomain = (-1,4)
  elif choice =="exponential":
    fun = exp_function
    domain = (-2,10)
    counterdomain = (-1,3)
  else:
    fun = polynomial_function
    domain = (-3,3)
    counterdomain = (-20,20)
  return fun, domain, counterdomain

def plot_originator_function(f, domain, counterdomain, xpoints, eps):
  # Obtain observation measurments with noise
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  ypoints = f(xpoints, eps)
  # Plot funcition with measurments
  maxloc = np.argmax(f(xgrid))
  maxval = xgrid[maxloc]
  maxfun = f(maxval)
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=xgrid,y=fgrid,mode='lines',line=dict(dash='dash'),name='Unknown function'))
  fig.add_trace(go.Scatter(x=xpoints,y=ypoints,mode='markers',marker=dict(size=10),name='Measurements'))
  fig.add_trace(go.Scatter(x=[maxval,maxval],y=[-50,maxfun],mode='lines',line=dict(color='red',dash='dash'),name=f'Location of Maximum ({maxval})'))
  fig.add_trace(go.Scatter(x=[-5,maxval],y=[maxfun,maxfun],mode='lines',line=dict(color='green',dash='dash'),name=f'Value at Maximum ({round(maxfun,3)})'))
  fig.update_yaxes(range=[counterdomain[0], counterdomain[1]])
  fig.update_xaxes(range=[domain[0],domain[1]])
  fig.update_layout(title="Unknown function with measurments",xaxis_title="x",yaxis_title="y")
  fig.show()

  return ypoints

def fit_gp_model(xpoints, ypoints):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-2, 1e1))
  gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, alpha=0, n_restarts_optimizer=3).fit(xpoints.reshape(-1,1), ypoints.reshape(-1,1))
  # Use model to make predictions
  xpred = xgrid
  ypred = np.zeros((3,len(xpred)))
  for n in range(len(xpred)):
      (mean,std)=gpr.predict(xgrid[n].reshape(-1,1),return_std=True)
      ypred[1,n] = mean[0]
      ypred[0,n] = mean[0]-2*std[0]
      ypred[2,n] = mean[0]+2*std[0]

  return ypred, gpr

def plot_model_predictions(f, domain, counterdomain, xpoints, ypoints, ypred):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  # Plot predictions
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[1,:],mode='lines', line=dict(width=2,color=px.colors.qualitative.G10[3]),name='Mean prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[0,:],mode='lines', line=dict(width=0,color=px.colors.qualitative.G10[3]),showlegend=False))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[2,:],mode='lines', line=dict(width=0,color=px.colors.qualitative.G10[3]),fill='tonexty',name='Std prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=fgrid,mode='lines',line=dict(dash='dash',color=px.colors.qualitative.G10[0]),name='Unknown function'))
  fig.add_trace(go.Scatter(x=xpoints,y=ypoints,mode='markers',marker=dict(size=10,color=px.colors.qualitative.G10[1]),name='Measurements'))
  fig.update_yaxes(range=[counterdomain[0], counterdomain[1]])
  fig.update_xaxes(range=[domain[0],domain[1]])
  fig.update_layout(title="Learned function from observations",xaxis_title="x",yaxis_title="y")
  fig.show()

def plot_exploration_explotation(f, domain, counterdomain, ypred, xpoints, ypoints, KAPPA1, KAPPA2):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  # Calculate EE (Explore-Exploit)
  eegrid1 = (1-KAPPA1)*ypred[1,:] + KAPPA1*((ypred[1,:]-ypred[0,:])/2)
  eegrid2 = (1-KAPPA2)*ypred[1,:] + KAPPA2*((ypred[1,:]-ypred[0,:])/2)
  # Plot acquisition function (EE)
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[1,:],mode='lines', line=dict(width=2,color='green'),name='Mean prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[0,:],mode='lines', line=dict(width=0,color='green'),showlegend=False))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[2,:],mode='lines', line=dict(width=0,color='green'),fill='tonexty',name='Std prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=fgrid,mode='lines',line=dict(dash='dash',color='blue'),name='Unknown function'))
  fig.add_trace(go.Scatter(x=xpoints,y=ypoints,mode='markers',marker=dict(size=10,color='red'),name='Measurements'))
  fig.add_trace(go.Scatter(x=xgrid,y=eegrid1,mode='lines',line=dict(color='black'),name=f'Explore-Exploit ({KAPPA1})'),secondary_y=True)
  fig.add_trace(go.Scatter(x=xgrid,y=eegrid2,mode='lines',line=dict(color='red'),name=f'Explore-Exploit ({KAPPA2})'),secondary_y=True)
  fig.update_yaxes(range=[counterdomain[0], 2*counterdomain[1]], secondary_y=False)
  fig.update_xaxes(range=[domain[0],domain[1]])
  fig.update_yaxes( secondary_y=True,title_text="Explore-Exploit")
  fig.update_layout(title="Acquisition function (Explore-Exploit) for all x",xaxis_title="x",yaxis_title="y")
  fig.show()

def plot_prob_improvement_1point(f, domain, counterdomain, xpoints, ypoints, xnew, gpr):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  # Obtain prediction at the new point
  (xnew_mean,xnew_std) = gpr.predict(np.array(xnew).reshape(1,-1),return_std=True)

  # Actual maximum value
  y_max = np.max(ypoints)

  # Compute normal distribution
  xdist = np.random.randn(100000)*float(xnew_std[0]) + float(xnew_mean[0])
  hist_data = [xdist]
  group_labels = ['Conditional Distribution']
  # Probability of improvement
  PI = 1-norm(float(xnew_mean[0]), float(xnew_std[0])).cdf(y_max)
  # Plot distribution
  fig = ff.create_distplot(hist_data, group_labels,curve_type='normal',show_hist=False, show_rug=False)
  fig.add_vline(x=float(xnew_mean[0]),line=dict(color='green'),annotation=dict(text="Mean",textangle=90))
  fig.add_vline(x=float(xnew_mean[0])+float(xnew_std[0]),line=dict(color='green',dash='dash'),annotation=dict(text="Std",textangle=90))
  fig.add_vline(x=float(xnew_mean[0])-float(xnew_std[0]),line=dict(color='green',dash='dash'),annotation=dict(text="Std",textangle=90))
  fig.add_vline(x=float(y_max),line=dict(color='red'),annotation=dict(xanchor="right",text="Observed Maximum Value",textangle=90))
  fig.add_trace(go.Scatter(x=[y_max,y_max,max(hist_data[0]),max(hist_data[0]),y_max],y=[0,max(fig.data[0]['y']),max(fig.data[0]['y']),0,0],mode='lines',line=dict(color='lightgoldenrodyellow'),name=f"Probability of Improvement {round(PI,3)}",fill='toself'))
  fig.update_layout(title="Conditional distribution at x = "+str(xnew),showlegend=True,xaxis_title="x",yaxis_title="pdf")
  fig.show()

def plot_prob_improvemnet(f, domain, counterdomain, ypred, xpoints, ypoints):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  y_max = np.max(ypoints)
  # Calculate PI (Probability of Improvement)
  pigrid = np.zeros_like(xgrid)
  for n in range(len(xgrid)):
      pigrid[n] = 1-norm(float(ypred[1,n]), float((ypred[1,n]-ypred[0,n])/2)).cdf(y_max)
  # Plot acquisition function (PI)
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[1,:],mode='lines', line=dict(width=2,color='green'),name='Mean prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[0,:],mode='lines', line=dict(width=0,color='green'),showlegend=False))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[2,:],mode='lines', line=dict(width=0,color='green'),fill='tonexty',name='Std prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=fgrid,mode='lines',line=dict(dash='dash',color='blue'),name='Unknown function'))
  fig.add_trace(go.Scatter(x=xpoints,y=ypoints,mode='markers',marker=dict(size=10,color='red'),name='Measurements'))
  fig.add_trace(go.Scatter(x=xgrid,y=pigrid,mode='lines',line=dict(color='red'),name='Probability of Improvement'),secondary_y=True)
  fig.update_yaxes(range=[counterdomain[0], counterdomain[1]], secondary_y=False)
  fig.update_xaxes(range=[domain[0],domain[1]])
  fig.update_yaxes(range=[0,1], secondary_y=True,title_text="Probability of Improvement")
  fig.update_layout(title="Acquisition function (Probability of Improvement) for all x",xaxis_title="x",yaxis_title="y")
  fig.show()

def plot_expected_improvement(f, domain, counterdomain, ypred, xpoints, ypoints):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  y_max = np.max(ypoints)
  # Calculate EI (Expected Improvement)
  eigrid = np.zeros_like(xgrid)
  xi = 0.1
  for n in range(len(xgrid)):
      sigma = float((ypred[1,n]-ypred[0,n])/2)
      zscore = (float(ypred[1,n])-y_max-xi) / sigma
      C = norm(0,1).cdf(zscore)
      P = norm(0,1).pdf(zscore)
      eigrid[n] = (float(ypred[1,n])-y_max-xi)*C + sigma*P

  # Plot acquisition function (EI)
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[1,:],mode='lines', line=dict(width=2,color='green'),name='Mean prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[0,:],mode='lines', line=dict(width=0,color='green'),showlegend=False))
  fig.add_trace(go.Scatter(x=xgrid,y=ypred[2,:],mode='lines', line=dict(width=0,color='green'),fill='tonexty',name='Std prediction'))
  fig.add_trace(go.Scatter(x=xgrid,y=fgrid,mode='lines',line=dict(dash='dash',color='blue'),name='Unknown function'))
  fig.add_trace(go.Scatter(x=xpoints,y=ypoints,mode='markers',marker=dict(size=10,color='red'),name='Measurements'))
  fig.add_trace(go.Scatter(x=xgrid,y=eigrid,mode='lines',line=dict(color='red'),name=f'Expected Improvement (xi={xi})'),secondary_y=True)
  fig.update_yaxes(range=[counterdomain[0], counterdomain[1]], secondary_y=False)
  fig.update_xaxes(range=[domain[0],domain[1]])
  fig.update_yaxes( secondary_y=True,title_text="Expected Improvement")
  fig.update_layout(title="Acquisition function (Expected Improvement) for all x",xaxis_title="x",yaxis_title="y")
  fig.show()

  # Define plotting functions
def posterior(optimizer, x_obs, y_obs, grid):
  optimizer._gp.fit(x_obs, y_obs)

  mu, sigma = optimizer._gp.predict(grid, return_std=True)
  return mu, sigma

def plot_gp(optimizer, domain):
  xgrid = np.linspace(domain[0],domain[1],1000)
  x = xgrid.reshape(-1, 1)
  y = f(xgrid,eps=0)
  fig = plt.figure(figsize=(16, 10))
  steps = len(optimizer.space)
  fig.suptitle(
      'Gaussian Process and Utility Function After {} Steps'.format(steps),
      fontdict={'size':30}
  )

  gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
  axis = plt.subplot(gs[0])
  acq = plt.subplot(gs[1])

  x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
  y_obs = np.array([res["target"] for res in optimizer.res])

  mu, sigma = posterior(optimizer, x_obs, y_obs, x)
  axis.plot(x, y, linewidth=3, label='Target')
  axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label=u'Observations', color='r')
  axis.plot(x, mu, '--', color='k', label='Prediction')

  axis.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
      alpha=.6, fc='c', ec='None', label='95% confidence interval')

  axis.set_xlim((domain[0], domain[1]))
  axis.set_ylim((None, None))
  axis.set_ylabel('f(x)', fontdict={'size':20})
  axis.set_xlabel('x', fontdict={'size':20})

  utility_function = UtilityFunction(kind="ucb", kappa=5, xi=0)
  utility = utility_function.utility(x, optimizer._gp, 0)
  acq.plot(x, utility, label='Utility Function', color='purple')
  acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15,
            label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
  acq.set_xlim((domain[0], domain[1]))
  acq.set_ylim((0, np.max(utility) + 0.5))
  acq.set_ylabel('Utility', fontdict={'size':20})
  acq.set_xlabel('x', fontdict={'size':20})

  axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
  acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def start_optimizer(domain):
  SEED = np.random.seed(42)
  xgrid = np.round(np.arange(domain[0],domain[1],0.01),4)
  fgrid = f(xgrid)
  # Plot defaults
  x = xgrid.reshape(-1, 1)
  y = f(xgrid,eps=0)
  # Define Optimization
  optimizer = BayesianOptimization(f, {'x': domain}, random_state=1,verbose=2)
  optimizer.maximize(init_points=2, n_iter=0)
  # Modify Model
  kernel=Matern(length_scale=1, nu=2.5, length_scale_bounds=(1e-3,1e3))
  optimizer._gp.set_params(**dict(alpha=0.1,kernel=kernel))

  return optimizer