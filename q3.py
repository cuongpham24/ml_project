import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import multivariate_normal
from functools import reduce 
data= sio.loadmat('data.mat')
data2=sio.loadmat('label.mat')
xtrain=data['data']
print(xtrain.shape)
ytrain= data2['trueLabel'].flatten()
print(ytrain.shape)
n=xtrain.shape[0]
print(n) # 1900 data point * 784 features

## center data
mean_vector = np.mean(xtrain,axis=1)
print(mean_vector.shape)
center_data= xtrain-mean_vector[:,np.newaxis]
center_data=center_data.T
print(center_data.shape)

## applying PCA using sklearn with original dataset, no centering
from sklearn.decomposition import PCA
mod=PCA(n_components=4)
model=mod.fit_transform(xtrain.T)
proj_data=model

## set up the model
dim=4
nummod=2
pi=np.random.randn(nummod) 
pi=pi/np.sum(pi) # this is prior likelihood of P(k) with k=2
print(pi)
## setting the mean, covariance
mean=np.random.randn(nummod,dim)
mean_old=mean.copy()
S1=np.random.randn(dim,dim)
cov1=np.matmul(S1,S1.T) + np.identity(dim)
S2=np.random.randn(dim,dim)
cov2=np.matmul(S2,S2.T) + np.identity(dim)
#print(cov1.shape)
var=[cov1,cov2]

## initialize the posterior - prob of (x|k)
tau=np.zeros((proj_data.shape[0],nummod))
print(tau.shape)
min=0.15 # min to converge
diff=min+1
i=0
## helper functiont 
def multiply(givenlist):
    time=1
    for each in givenlist:
        time *=each
    return time 
##
while diff > min: 
    print("num of inter", i )
    ## generaing G model
    G1=multivariate_normal(mean[0],cov1)
    G2=multivariate_normal(mean[1],cov2)
    ## weighted-pdf for 2 model
    sump= (multivariate_normal.pdf(proj_data,mean_old[0],cov1) * pi[0]) + (multivariate_normal.pdf(proj_data,mean_old[1],cov2) * pi[1])
    log_lihood = multivariate_normal.pdf(proj_data,mean_old[0],cov1) + multivariate_normal.pdf(proj_data,mean_old[1],cov2)  # Likelihood at each given data point P(X|G=1) + P(X|G=2)
    #print(sump.shape) # (1990,)
    pn1=np.divide((multivariate_normal.pdf(proj_data,mean_old[0],cov1) * pi[0]),sump) # posterior p(G=1|X)
    #print(pn1.shape)
    #print(proj_data.shape)
    pn2=np.divide((multivariate_normal.pdf(proj_data,mean_old[1],cov2) * pi[1]),sump) # Posterior p(G=2|X)
    #mean= np.sum(np.multiply(proj_data,pn1[:,None]) + np.multiply(proj_data,pn2[:,None]),axis=0)/ np.sum(pn1+pn2,axis=0) ## new overall mean
    #print(mean.shape)

    ## loglihood
    mean_old[0]=mean[0]
    mean_old[1]=mean[1]
    mean[0]= np.sum(np.multiply(proj_data,pn1[:,None]),axis=0)/ np.sum(pn1,axis=0)
    mean[1]= np.sum(np.multiply(proj_data,pn2[:,None]),axis=0)/np.sum(pn2,axis=0)
    #print(mean_old[0],mean_old[1]) ## it is new mean for each model
    ## updating covariance
    cov1=np.dot((np.multiply((proj_data-mean_old[0]),pn1[:,None]).T),np.multiply((proj_data-mean_old[0]),pn1[:,None]))/np.sum(pn1,axis=0)
    cov2=np.dot((np.multiply((proj_data-mean_old[1]),pn2[:,None]).T),np.multiply((proj_data-mean_old[1]),pn2[:,None]))/np.sum(pn2,axis=0)
    # updating prior
    pi[0]= np.sum(pn1,axis=0)/proj_data.shape[0]
    pi[1]= np.sum(pn2,axis=0)/proj_data.shape[0]
    # condition to converge
    diff= np.linalg.norm((mean-mean_old),ord=2)
    print("changes in mean ", diff)
    print("log",(np.log(log_lihood)))
    if np.where(log_lihood==0):
        print("zero exist")
    i=i+1
    log=1
#if diff < min:
#    print("converge")
#    break
#if i >100:
#    print("max allow")
#    break