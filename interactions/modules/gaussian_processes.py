import numpy as np
import pandas as pd
import copy
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED = 42

""" Originator function definition """
def f(x, beta0, beta1, beta2, beta3, sigma):
    return np.matmul(np.vstack((np.ones(len(x)), np.array(x), np.array(x) **2, np.array(x) **3)).T,np.array([beta0,beta1,beta2,beta3])) + sigma*np.random.normal(0,1,len(x))

""" Kernel function definition """
def kernel_fcn(xi,xj,sigma=2, kernel="RBF"):
  if kernel.lower() == "rbf":
    ## This is rbf / normal distribution kernel
    return np.exp(-0.5*((np.array(xi)-np.array(xj))/sigma)**2)
  elif kernel == "laplacian":
    ## This is laplacian kernel
    return np.exp(- sigma*np.abs((np.array(xi) - np.array(xj))))
  elif kernel == "polynomial":
    ## This is polynomial kernel
    return (np.array(xi)*np.array(xj)+1)**sigma
  else:
    print("Invalid kernel choice. Default to RBF kernel.")
    return np.exp(-0.5*((np.array(xi)-np.array(xj))/sigma)**2)

""" Plot Generator Function"""
def plot_generator_function(XGRID=np.arange(0,7,0.1), XPOINTS=np.array(range(1,7)), beta0=6, beta1=-3, beta2=0.5, beta3=0.0, sigma=0.2, kernel = None, kernel_center = None, kernel_sigma = None):
  np.random.seed(SEED)
  YPOINTS = f(XPOINTS,beta0,beta1,beta2,beta3,sigma)
  fig = make_subplots(specs=[[{"secondary_y": True}]])
  fig.add_trace(go.Scatter(x=XGRID,y=f(XGRID,beta0,beta1,beta2,beta3,sigma=0),mode='lines',line=dict(color = 'rgb(49,130,189)'),name="Originator function"))
  fig.add_trace(go.Scatter(x=XPOINTS,y=YPOINTS,mode='markers',marker=dict(color = 'rgb(239,85,59)',size=10),name="Generated data"))
  if kernel is not None:
    fig.add_trace(go.Scatter(x=XGRID,y=kernel_fcn(XGRID,kernel_center,kernel_sigma,kernel=kernel),mode='lines',marker=dict(size=10),name="Kernel function"),secondary_y=True)
  fig.update_layout(showlegend=True,title="Generated function we try to learn",xaxis_title="x",yaxis_title="y")
  fig.update_layout(width=1000)
  fig.show()
  return YPOINTS

""" Plot Generated Data"""
def plot_generated_experiments(NO_EXP=25, XPOINTS=np.array(range(1,7)), beta0=6, beta1=-3, beta2=0.5, beta3=0.0, sigma=0.2):
  np.random.seed(SEED)
  YEXP = np.zeros((NO_EXP,len(XPOINTS)))
  fig = px.scatter()
  for n in range(NO_EXP):
      YEXP[n,:] = f(XPOINTS,float(np.random.normal(size=1)[0]*0.5+beta0),float(np.random.normal(size=1)[0]*0.1+beta1),float(np.random.normal(size=1)[0]*0.01+beta2),float(np.random.normal(size=1)[0]*0.0005+beta3),sigma=0.2)
      fig.add_trace(go.Scatter(x=XPOINTS,y=YEXP[n,:],mode='lines+markers',line=dict(color = 'rgb(49,130,189)')))
  fig.update_layout(showlegend=False,title="Generated data",xaxis_title="x",yaxis_title="y")
  fig.update_layout(width=1000)
  fig.show()
  return YEXP

def plot_correlation_steps(YEXP, STEP_XAXIS = 1, STEP_YAXIS = 2):
  fig = px.scatter(x=YEXP[:,STEP_XAXIS-1],y=YEXP[:,STEP_YAXIS-1],
                 title = "Correlation between step x = "+str(STEP_XAXIS)+" and x = "+str(STEP_YAXIS)+" with correlation being "+str(np.round(np.corrcoef(YEXP[:,STEP_XAXIS-1],YEXP[:,STEP_YAXIS-1])[0,1],3)),
                 labels=dict(x="Y(x = "+str(STEP_XAXIS)+")", y="Y(x = "+str(STEP_YAXIS)+")"),
                 marginal_x="box", marginal_y="box",)
  fig.update_layout(width=1000,height=700)
  fig.show()

def multivariate_normal_distribution(YEXP, XPOINTS):
  Y_mu = np.mean(YEXP,axis=0)
  Y_cov = np.cov(YEXP,rowvar=False)
  Y_corr = np.corrcoef(YEXP,rowvar=False)

  print('Mean: \n',pd.DataFrame(Y_mu,index=XPOINTS).T)
  print('\nCovariance: \n',pd.DataFrame(Y_cov,index=XPOINTS,columns=XPOINTS))
  print('\nCorrelation: \n',pd.DataFrame(Y_corr,index=XPOINTS,columns=XPOINTS))

  YMN = np.random.multivariate_normal(Y_mu,Y_cov,100000)
  return YMN

def conditional_probability_1step(YMN, STEP_A = 1, VALUE_A= 3, STEP_B = 2, TOL = 0.01, NUM_SIM = 20):
  # Match samples from the distribution that specify observations above
  rows = np.logical_and(YMN[:,STEP_A-1] > VALUE_A - TOL,YMN[:,STEP_A-1] > VALUE_A + TOL)
  # Compute conditional probability exactly
  mu_step = np.mean(YMN[rows,STEP_B-1])
  std_step = np.std(YMN[rows,STEP_B-1])
  print('\n Conditional Mean: ',mu_step)
  print('\n Conditional Deviation: ',std_step,'\n')
  # Plot histogram
  fig = px.histogram(YMN[rows,STEP_B-1],histnorm='probability density')
  fig.update_layout(bargap=0.01,showlegend=False,title="Conditional Probability",xaxis_title="P(x="+str(STEP_B)+"|f(x="+str(STEP_A)+")="+str(VALUE_A)+")",yaxis_title="Probability density")
  fig.add_vline(mu_step)
  fig.add_vline(mu_step-std_step,line=dict(color='orange'))
  fig.add_vline(mu_step+std_step,line=dict(color='orange'))
  fig.add_vline(mu_step-2*std_step,line=dict(color='orange',dash='dash'))
  fig.add_vline(mu_step+2*std_step,line=dict(color='orange',dash='dash'))
  fig.show()

  fig = px.scatter()
  for n in range(NUM_SIM):
      fig.add_trace(go.Scatter(x=[STEP_A,STEP_B],y=[VALUE_A,YMN[rows,STEP_B-1][n]],mode='lines+markers',line=dict(color = 'rgb(49,130,189)')))
  fig.add_trace(go.Scatter(x=[STEP_B,STEP_B],y=[mu_step-2*std_step,mu_step+2*std_step],mode='lines+markers',line=dict(color = px.colors.qualitative.G10[1])))
  fig.add_trace(go.Scatter(x=[STEP_B,STEP_B],y=[mu_step,mu_step],mode='markers',marker=dict(size=10,color = px.colors.qualitative.G10[1])))
  fig.add_trace(go.Scatter(x=[STEP_A,STEP_A],y=[VALUE_A,VALUE_A],mode='markers',marker=dict(size=15,color = 'rgb(49,190,130)')))
  fig.update_layout(showlegend=False,title="Conditional Probability given observations",xaxis_title="x",yaxis_title="y")
  fig.update_xaxes(range=[STEP_A-1,STEP_B+1])
  fig.show()

def conditional_probability_multiple_steps(YMN, STEP_A = 1, VALUE_A= 3, STEPS_OTHER = [2,3,4,5,6], TOL = 0.01, NUM_SIM = 40):
  STEPS_ALL = copy.copy(STEPS_OTHER)
  STEPS_ALL.insert(0,STEP_A)
  rows = np.logical_and(YMN[:,STEP_A-1] > VALUE_A - TOL,YMN[:,STEP_A-1] > VALUE_A + TOL)
  mu_OTHER = []
  std_OTHER = []
  for i in STEPS_OTHER:
      mu_OTHER.append(np.mean(YMN[rows,i-1]))
      std_OTHER.append(np.std(YMN[rows,i-1]))
  df_sum = pd.DataFrame(columns = ["steps","means"])
  df_sum["steps"]= STEPS_OTHER
  df_sum["means"]= mu_OTHER
  print('\nMeans: \n',df_sum.to_string(index=False))
  Y_corr = np.corrcoef(YMN[rows][:,[i-1 for i in STEPS_OTHER]],rowvar=False)
  print('\nCorrelations: \n',pd.DataFrame(Y_corr,columns=STEPS_OTHER,index=STEPS_OTHER),'\n')
  fig = go.Figure()
  for i in STEPS_OTHER:
      fig.add_trace(go.Histogram(x=YMN[rows,i-1],histnorm='probability density',name="x = "+str(i)))
  # Overlay both histograms
  fig.update_layout(barmode='overlay')
  # Reduce opacity to see both histograms
  fig.update_traces(opacity=0.75)
  fig.update_layout(bargap=0.01,showlegend=True,title="Conditional Probability at all steps",xaxis_title="P(x=" +str(STEPS_OTHER)+" | f(x="+str(STEP_A)+")="+str(VALUE_A)+")",yaxis_title="Probability density")
  fig.show()

  fig = px.scatter()
  for n in range(NUM_SIM):
      fig.add_trace(go.Scatter(x=STEPS_ALL,y=np.concatenate((np.array(VALUE_A),YMN[rows][:,[i-1 for i in STEPS_OTHER]][n]),axis=None),mode='lines+markers',line=dict(color = 'rgb(49,130,189)')))
  fig.add_trace(go.Scatter(x=[STEP_A,STEP_A],y=[VALUE_A,VALUE_A],mode='markers',marker=dict(size=15,color = 'rgb(49,190,130)')))
  for i,s in enumerate(STEPS_OTHER):
      fig.add_trace(go.Scatter(x=[s,s],y=[mu_OTHER[i]],mode='markers',marker=dict(size=10,color = px.colors.qualitative.G10[1])))
      fig.add_trace(go.Scatter(x=[s,s],y=[mu_OTHER[i]-2*std_OTHER[i],mu_OTHER[i]+2*std_OTHER[i]],mode='lines+markers',line=dict(color = px.colors.qualitative.G10[1])))
  fig.update_layout(showlegend=False,title="Conditional Probability given observations",xaxis_title="x",yaxis_title="y")
  fig.update_xaxes(range=[STEPS_ALL[0]-1,STEPS_ALL[-1]+1])
  fig.show()

def conditional_probability_multiple_observations(YMN, STEP_A = 1, VALUE_A= 3, STEP_B = 4, VALUE_B= 2, STEPS_OTHER = [2,3,5,6], TOL = 0.01, NUM_SIM = 40):
  STEPS_ALL = copy.copy(STEPS_OTHER)
  STEPS_ALL.append(STEP_A)
  STEPS_ALL.append(STEP_B)
  STEPS_ALL.sort()
  rows = np.logical_and(np.logical_and(YMN[:,STEP_A-1] > VALUE_A - TOL,YMN[:,STEP_A-1] > VALUE_A + TOL),np.logical_and(YMN[:,STEP_B-1] > VALUE_B - TOL,YMN[:,STEP_B-1] > VALUE_B + TOL))
  mu_OTHER = []
  std_OTHER = []
  for i in STEPS_OTHER:
      mu_OTHER.append(np.mean(YMN[rows,i-1]))
      std_OTHER.append(np.std(YMN[rows,i-1]))
  df_sum = pd.DataFrame(columns = ["steps","means"])
  df_sum["steps"]= STEPS_OTHER
  df_sum["means"]= mu_OTHER
  print('\nMeans: \n',df_sum.to_string(index=False))
  Y_corr = np.corrcoef(YMN[rows][:,[i-1 for i in STEPS_OTHER]],rowvar=False)
  print('\nCorrelations: \n',pd.DataFrame(Y_corr,columns=STEPS_OTHER,index=STEPS_OTHER),'\n')
  fig = go.Figure()
  for i in STEPS_OTHER:
      fig.add_trace(go.Histogram(x=YMN[rows,i-1],histnorm='probability density',name="x = "+str(i)))
  # Overlay both histograms
  fig.update_layout(barmode='overlay')
  # Reduce opacity to see both histograms
  fig.update_traces(opacity=0.75)
  fig.update_layout(bargap=0.01,showlegend=True,title="Conditional Probability at all steps",xaxis_title="P(x=" +str(STEPS_OTHER)+" | f(x="+str(STEP_A)+")="+str(VALUE_A)+"), f(x="+str(STEP_B)+")="+str(VALUE_B)+")",yaxis_title="Probability density")
  fig.show()

  fig = px.scatter()
  for n in range(NUM_SIM):
      values_all = np.concatenate((np.array(VALUE_A),YMN[rows][:,[i-1 for i in STEPS_OTHER[0:-2]]][n],np.array(VALUE_B),YMN[rows][:,[i-1 for i in STEPS_OTHER[-2:]]][n]),axis=None)
      fig.add_trace(go.Scatter(x=STEPS_ALL,y=values_all,mode='lines+markers',line=dict(color = 'rgb(49,130,189)')))
  fig.add_trace(go.Scatter(x=[STEP_A,STEP_A],y=[VALUE_A,VALUE_A],mode='markers',marker=dict(size=15,color = 'rgb(49,190,130)')))
  fig.add_trace(go.Scatter(x=[STEP_B,STEP_B],y=[VALUE_B,VALUE_B],mode='markers',marker=dict(size=15,color = 'rgb(49,190,130)')))
  for i,s in enumerate(STEPS_OTHER):
      fig.add_trace(go.Scatter(x=[s,s],y=[mu_OTHER[i]],mode='markers',marker=dict(size=10,color = px.colors.qualitative.G10[1])))
      fig.add_trace(go.Scatter(x=[s,s],y=[mu_OTHER[i]-2*std_OTHER[i],mu_OTHER[i]+2*std_OTHER[i]],mode='lines+markers',line=dict(color = px.colors.qualitative.G10[1])))
  fig.update_layout(showlegend=False,title="Conditional Probability given observations",xaxis_title="x",yaxis_title="y")
  fig.update_xaxes(range=[STEPS_ALL[0]-1,STEPS_ALL[-1]+1])
  fig.show()

def mvncond2(XPOINTS, YPOINTS, error, k, kernel_sigma):
    joint_mean = np.zeros(len(XPOINTS)+1) # joint_mean is a zero vector
    joint_cov = kernel_sigma*k + np.diag(np.diag(error*np.ones((len(XPOINTS)+1,len(XPOINTS)+1))))
    i = len(XPOINTS) # index pointing to a new obs in covariance matrix
    obs_f = YPOINTS
    obs_mu= np.mean(YPOINTS)
    obs_diff = obs_f - obs_mu # obs_diff = obs_f - obs_mu de-meaned predictions
    n = joint_cov.shape[0]
    kxx = joint_cov[0:i,0:i]
    kxsx = joint_cov[i,0:i]
    s12s22 = np.matmul(kxsx,np.linalg.inv(kxx))
    cond_mean =joint_mean[i] +  np.matmul(s12s22,(obs_diff-joint_mean[i:n]))
    cond_cov = joint_cov[i,i] - np.matmul(s12s22,np.transpose(kxsx))

    new_obs_mu = cond_mean + obs_mu
    new_obs_std = np.sqrt(cond_cov)
    return new_obs_mu, new_obs_std

def kernel_matrix(XPOINTS, XPOINTS_NEW, kernel, kernel_sigma, visualize_kernel=False):
   # kernel matrix from given observations
  no_data = len(XPOINTS)
  k = np.zeros((no_data+1,no_data+1));
  for i in range(no_data):
    k[i,:len(XPOINTS)] = kernel_fcn(XPOINTS[i],XPOINTS,kernel_sigma, kernel)
  # update kernel matrix with new observation
  x_pred = np.array(XPOINTS_NEW)
  x_new = np.append(XPOINTS,x_pred)
  k[:,len(XPOINTS)] =  kernel_fcn(x_pred,x_new,kernel_sigma, kernel)
  k[len(XPOINTS),:] =  kernel_fcn(x_pred,x_new,kernel_sigma, kernel)

  # Visualize kernel
  if visualize_kernel:
    K_pred = pd.DataFrame(k); K_pred.index = x_new ; K_pred.columns = x_new
    print("\nKernel given new observations: \n",K_pred)
  
  return k



def multivariate_distribution_1point(XPOINTS, YPOINTS, error=0.0, XPOINTS_NEW=6, XGRID=np.arange(0,12,0.1), beta0=6, beta1=-3, beta2=0.5, beta3=0.0, sigma=0.2, kernel = "RBF", kernel_sigma=2.0):
  k= kernel_matrix(XPOINTS, XPOINTS_NEW, kernel, kernel_sigma, visualize_kernel=True)

  # Calculate the conditional mean and std
  error=sigma
  new_obs_mu, new_obs_std = mvncond2(XPOINTS,YPOINTS,error,k,kernel_sigma)
  print('\n Conditional mean of new obs: ',round(new_obs_mu,4))
  print('\n Conditional std of new obs: ',round(new_obs_std,4),"\n")

  # Plot originator function with generated data
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=XGRID,y=f(XGRID,beta0,beta1,beta2,beta3,sigma=0),mode='lines',name="Originator function"))
  fig.add_trace(go.Scatter(x=XPOINTS,y=YPOINTS,mode='markers',marker=dict(size=10),name="Generated data"))
  fig.add_trace(go.Scatter(x=[XPOINTS_NEW,XPOINTS_NEW],y=[new_obs_mu-2*new_obs_std,new_obs_mu+2*new_obs_std],mode='markers+lines',line=dict(color = px.colors.qualitative.G10[3]),name = "Conditional Deviation"))
  fig.add_trace(go.Scatter(x=[XPOINTS_NEW,XPOINTS_NEW],y=[new_obs_mu,new_obs_mu],mode='markers',marker=dict(size=10,color = px.colors.qualitative.G10[3]),name = "Conditional Mean"))
  fig.update_layout(showlegend=True,title="Conditional prediction at a new point x="+str(XPOINTS_NEW),xaxis_title="x",yaxis_title="y")
  fig.show()


def multivariate_distribution_all_points(XPOINTS, YPOINTS, XGRID=np.arange(0,12,0.1), beta0=6, beta1=-3, beta2=0.5, beta3=0.0, sigma=0.2, kernel = "RBF", kernel_sigma=2.0):
  xgrid_mu = []
  xgrid_std = []
  for new_obs_point in XGRID:
    k= kernel_matrix(XPOINTS, new_obs_point, kernel, kernel_sigma)
    error=sigma
    new_obs_mu, new_obs_std = mvncond2(XPOINTS,YPOINTS,error,k,kernel_sigma)
    xgrid_mu.append(new_obs_mu)
    xgrid_std.append(new_obs_std)

  fig = go.Figure()
  fig.add_trace(go.Scatter(x=XGRID,y=f(XGRID,beta0,beta1,beta2,beta3,sigma=0),mode='lines',name="Originator function"))
  fig.add_trace(go.Scatter(x=XPOINTS,y=YPOINTS,mode='markers',marker=dict(size=10),name="Generated data"))
  fig.add_trace(go.Scatter(x=XGRID,y=xgrid_mu,mode='lines',line=dict(width=3,color = px.colors.qualitative.G10[3]),name = "Conditional Mean"))
  fig.add_trace(go.Scatter(x=XGRID,y=np.array(xgrid_mu)-2*np.array(xgrid_std),mode='lines',line=dict(width=1,color = px.colors.qualitative.G10[3]),showlegend=False))
  fig.add_trace(go.Scatter(x=XGRID,y=np.array(xgrid_mu)+2*np.array(xgrid_std),mode='lines',fill='tonexty',line=dict(width=1,color = px.colors.qualitative.G10[3]),name = "Conditional Deviation"))

  fig.update_layout(showlegend=True,title="Conditional prediction at all points",xaxis_title="x",yaxis_title="y")
  fig.show()
