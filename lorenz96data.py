#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data
"""
import numpy as np
from numpy.random import seed
seed(1)
from scipy import integrate
from scipy import linalg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

import matplotlib as mpl

def rhs(ne,u,fr):
    v = np.zeros(ne+3)
    v[2:ne+2] = u
    v[1] = v[ne+1]
    v[0] = v[ne]
    v[ne+2] = v[2]
    
    r = np.zeros(ne)
    
#    for i in range(2,ne+2):
#        r[i-2] = v[i-1]*(v[i+1] - v[i-2]) - v[i] + fr
    
    r = v[1:ne+1]*(v[3:ne+3] - v[0:ne]) - v[2:ne+2] + fr
    
    return r
    
    
def rk4(ne,dt,u,fr):
    r1 = rhs(ne,u,fr)
    k1 = dt*r1
    
    r2 = rhs(ne,u+0.5*k1,fr)
    k2 = dt*r2
    
    r3 = rhs(ne,u+0.5*k2,fr)
    k3 = dt*r3
    
    r4 = rhs(ne,u+k3,fr)
    k4 = dt*r4
    
    un = u + (k1 + 2.0*(k2 + k3) + k4)/6.0
    
    return un
       
#%%
ne = 40

dt = 0.005
tmax = 10.0
tini = 5.0
ns = int(tini/dt)
nt = int(tmax/dt)
fr = 10.0
nf = 10         # frequency of observation
nb = int(nt/nf) # number of observation time

u = np.zeros(ne)
utrue = np.zeros((ne,nt+1))
uinit = np.zeros((ne,ns+1))

#-----------------------------------------------------------------------------#
# generate true solution trajectory
#-----------------------------------------------------------------------------#
ti = np.linspace(-tini,0,ns+1)
t = np.linspace(0,tmax,nt+1)
tobs = np.linspace(0,tmax,nb+1)
x = np.linspace(1,ne,ne)

X,T = np.meshgrid(x,t,indexing='ij')
Xi,Ti = np.meshgrid(x,ti,indexing='ij')

u[:] = fr
u[int(ne/2)-1] = fr + 0.01
uinit[:,0] = u

# generate initial condition at t = 0
for k in range(1,ns+1):
    un = rk4(ne,dt,u,fr)
    uinit[:,k] = un
    u = np.copy(un)

# assign inital condition
u = uinit[:,-1]
utrue[:,0] = uinit[:,-1]

# generate true forward solution
for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    utrue[:,k] = un
    u = np.copy(un)
vmin = -12
vmax = 12
fig, ax = plt.subplots(2,1,figsize=(8,4))
cs = ax[0].contourf(Ti,Xi,uinit,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(uinit)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

#cs = ax[1].contourf(T,X,ufort_e,cmap='jet',vmin=-vmin,vmax=vmax)
#m = plt.cm.ScalarMappable(cmap='jet')
#m.set_array(uinit)
#m.set_clim(vmin, vmax)
#fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))

cs = ax[1].contourf(T,X,utrue,cmap='jet',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='jet')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))

fig.tight_layout()
plt.show()

#-----------------------------------------------------------------------------#
# generate observations
#-----------------------------------------------------------------------------#
mean = 0.0
sd2 = 1.0e-1 # added noise (variance)
sd1 = np.sqrt(sd2) # added noise (standard deviation)

oib = [nf*k for k in range(nb+1)]

uobs = utrue[:,oib] + np.random.normal(mean,sd1,[ne,nb+1])

#-----------------------------------------------------------------------------#
# generate erroneous soltions trajectory
#-----------------------------------------------------------------------------#
uw = np.zeros((ne,nt+1))
k = 0
si2 = 1.0e-2
si1 = np.sqrt(si2)

u = utrue[:,0] + np.random.normal(mean,si1,ne)
uw[:,0] = u

for k in range(1,nt+1):
    un = rk4(ne,dt,u,fr)
    uw[:,k] = un
    u = np.copy(un)

#%%
#-----------------------------------------------------------------------------#
# EnKF model
#-----------------------------------------------------------------------------#    

# number of observation vector
me = 4
freq = int(ne/me)
oin = [freq*i-1 for i in range(1,me+1)]

# number of ensemble 
npe = 40
cn = 1.0/np.sqrt(npe-1)

z = np.zeros((me,nb+1))
zf = np.zeros((me,npe,nb+1))
DhX = np.zeros((me,npe))
DhXm = np.zeros(me)

ua = np.zeros((ne,nt+1)) # mean analyssi solution (to store)
uf = np.zeros(ne)        # mean forecast
sc = np.zeros((ne,npe))   # square-root of the covariance matrix
ue = np.zeros((ne,npe,nt+1)) # all ensambles
ph = np.zeros((ne,me))

km = np.zeros((ne,me))
kmd = np.zeros((ne,npe))

cc = np.zeros((me,me))
ci = np.zeros((me,me))

for k in range(nb+1):
    z[:,k] = uobs[oin,k]
    for n in range(npe):
        zf[:,n,k] = z[:,k] + np.random.normal(mean,sd1,me)

# initial ensemble
k = 0
se2 = np.sqrt(sd2)
se1 = np.sqrt(se2)

for n in range(npe):
    ue[:,n,k] = uw[:,k] + np.random.normal(mean,sd1,ne)       
    
ua[:,k] = np.sum(ue[:,:,k],axis=1)
ua[:,k] = ua[:,k]/npe

kobs = 1

# RK4 scheme
for k in range(1,nt+1):
    
    # forecast afor all ensemble fields
    for n in range(npe):
        u[:] = ue[:,n,k-1]
        un = rk4(ne,dt,u,fr)
        ue[:,n,k] = un[:] + np.random.normal(mean,se1,ne)
    
    if k == oib[kobs]:
        # compute mean of the forecast fields
        uf[:] = np.sum(ue[:,:,k],axis=1)   
        uf[:] = uf[:]/npe
        
        # compute square-root of the covariance matrix
        for n in range(npe):
            sc[:,n] = cn*(ue[:,n,k] - uf[:])
        
        # compute DhXm data
        DhXm[:] = np.sum(ue[oin,:,k],axis=1)    
        DhXm[:] = DhXm[:]/npe
        
        # compute DhM data
        for n in range(npe):
            DhX[:,n] = cn*(ue[oin,n,k] - DhXm[:])
            
        # R = sd2*I, observation m+atrix
        cc = DhX @ DhX.T
        
        for i in range(me):
            cc[i,i] = cc[i,i] + sd2
        
        ph = sc @ DhX.T
                    
        ci = np.linalg.pinv(cc) # ci: inverse of cc matrix
        
        km = ph @ ci
        
        # analysis update    
        kmd = km @ (zf[:,:,kobs] - ue[oin,:,k])
        ue[:,:,k] = ue[:,:,k] + kmd[:,:]
        
        kobs = kobs+1
    
    # mean analysis for plotting
    ua[:,k] = np.sum(ue[:,:,k],axis=1)
    ua[:,k] = ua[:,k]/npe
    
#%%
fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

n = [0,14,34]
for i in range(3):
    ax[i].plot(tobs,uobs[n[i],:],'ro', lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')

    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=4, labelspacing=0.)
fig.tight_layout()
plt.show() 
fig.savefig('m_'+str(me)+'.pdf')

#%%
vmin = -10
vmax = 10
fig, ax = plt.subplots(2,1,figsize=(8,5))

cs = ax[0].contourf(T,X,utrue,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(utrue)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[0],ticks=np.linspace(vmin, vmax, 6))
ax[0].set_title('True')

cs = ax[1].contourf(T,X,ua,cmap='coolwarm',vmin=vmin,vmax=vmax)
m = plt.cm.ScalarMappable(cmap='coolwarm')
m.set_array(ua)
m.set_clim(vmin, vmax)
fig.colorbar(m,ax=ax[1],ticks=np.linspace(vmin, vmax, 6))
ax[1].set_title('EnKF')

fig.tight_layout()
plt.show() 
fig.savefig('f_'+str(me)+'.pdf')    
    

#%%    
uwe = np.zeros((ne,npe,nt+1)) # all ensambles
k = 0
for n in range(npe):
    uwe[:,n,k] = uw[:,k] + np.random.normal(mean,sd1,ne)

for k in range(1,nt+1):
    for n in range(npe):
        u[:] = uwe[:,n,k-1]
        un = rk4(ne,dt,u,fr)
        uwe[:,n,k] = un[:]

np.savez('data_sparse.npz',utrue=utrue, uobs=uobs, uwe = uwe)
        
fig, ax = plt.subplots(3,1,sharex=True,figsize=(10,6))

n = [0,19,39]
for i in range(3):
    ax[i].plot(tobs,uobs[n[i],:],'ro',lw=3)
    ax[i].plot(t,utrue[n[i],:],'k-')
    ax[i].plot(t,uw[n[i],:],'b--')
    ax[i].plot(t,ua[n[i],:],'g-.')
    ax[i].plot(t,uwe[n[i],0,:],'m--')
    
    ax[i].set_xlim([0,tmax])
    ax[i].set_ylabel(r'$x_{'+str(n[i]+1)+'}$')

ax[i].set_xlabel(r'$t$')
line_labels = ['Observation','True','Wrong','EnKF','LSTM Wrong J = 0']
plt.figlegend( line_labels,  loc = 'lower center', borderaxespad=-0.2, ncol=5, labelspacing=0.)
fig.tight_layout()
plt.show() 
#fig.savefig('m_'+str(me)+'.pdf')        

import pandas as pd
#观测数据保存
df1 = pd.DataFrame(tobs)
df2 = pd.DataFrame(uobs)
df2=df2.T
df = pd.concat([df1,df2],axis=1)
df.to_csv('osbdata.csv')
#真值数据保存
df1 = pd.DataFrame(t)
df2 = pd.DataFrame(utrue)
df2=df2.T
df = pd.concat([df1,df2],axis=1)
df.to_csv('truedata.csv')
#误数据保存
df1 = pd.DataFrame(t)
df2 = pd.DataFrame(uw)
df2=df2.T
df = pd.concat([df1,df2],axis=1)
df.to_csv('uwdata.csv')
#集合卡尔曼滤波数据保存
df1 = pd.DataFrame(t)
df2 = pd.DataFrame(ua)
df2=df2.T
df = pd.concat([df1,df2],axis=1)
df.to_csv('uadata.csv')


import pandas as pd
import numpy as np

# 假设var1和var2是两个3D数组
var1 = uwe
# 将3D数组展开为2D数组，形状为(40*400, 2001)
var1_reshaped = var1.reshape(-1, var1.shape[-1])

# 将其转换为DataFrame
df_var1 = pd.DataFrame(var1_reshaped)

# 保存到CSV文件
df_var1.to_csv('uwedata.csv', index=False)






















































