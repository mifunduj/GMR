import numpy as np
from scipy.signal import gaussian
from sklearn.mixture import GaussianMixture as GM
from matplotlib import pyplot as plt
from operator import itemgetter
import math
from math import exp, sqrt, pi
import csv

# Convert data to float
# Ref: joint_position_file_playback.py by Rethink Robotics
def try_float(x):
    try:
        return float(x)
    except ValueError:
        return None

# Order the data according to the increasing time
def ordered(x, n_col=17):
    x=sorted(x,key=itemgetter(0))
    l=len(x)
    a=np.zeros([l,n_col])
    for i in range(l):
        a[i,:]=[try_float(j) for j in x[i]]
    return a

# Read file
def read_file(_file_name, n_col=17):
    with open(_file_name,'r') as f:
        f_reader =csv.reader(f, delimiter=' ')
        next(f_reader)
        i=0
        lines=[]
        for line in f:
            i=i+1
            lines.append(line)
        n_row=i
        data=np.zeros([n_row,n_col])
        for i in range(n_row):
            data[i,:]=[try_float(x) for x in lines[i].rstrip().split(',')]
    return data,n_row

# Write file
_header=['time','left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2','left_gripper',
                 'right_s0','right_s1','right_e0','right_e1','right_w0','right_w1','right_w2','right_gripper']
def write_file(file_name, data,_header_=_header):
    n_row,n_col=data.shape
    with open(file_name, 'w') as g:
        _header_=_header
        for _name in _header_:
            g.write(str(_name) + ',')
        g.write('\n')
        for i in range(n_row):
            for j in range(n_col):
                s=str(data[i,j])
                if j==n_col-1:
                    g.write(s + '\n')
                else:
                    g.write(s + ',')
    print("%s file has been written" %file_name)

# Fuse data
def fuse_data(file,n=2,n_col=17):
    data_list=[]
    n_list=[]
    n_stamp_list=0
    n_begin=0
    n_end=0
    for i in range(1,n+1):
        s=str(i)
        stamp,n_stamp=read_file(file + s )
        data_list.append(stamp)
        n_list.append(n_stamp)
        n_stamp_list=n_stamp_list+n_stamp
        del stamp
    data=np.zeros([n_stamp_list,n_col])
    for j in range(n):
        n_post=n_list[j]
        n_begin=n_end
        n_end=n_begin+n_post
        for i in range(n_begin,n_end):
            data[i,:]=[try_float(x) for x in data_list[j][i-n_begin]]
    data=ordered(data)
    return data

def normalize_fuse_data(file,n=2,n_col=17):
# Normalise and fuse data
# Same parameters and return as fuse_data
    dict_data = dict()
    max_time_list=[]
    data_list=[]
    n_list=[]
    n_stamp_list=0
    n_begin=0
    n_end=0
    for i in range(n):
        dict_data["data_list_" +str(i+1)] =[]
    for i in range(1,n+1):
        s=str(i)
        dict_data["data_list_" +str(i)],n_stamp=read_file(file + s )
        n_list.append(n_stamp)
        n_stamp_list=n_stamp_list+n_stamp
    for name in dict_data.keys():
        max_time=(dict_data[name][-1])[0]
        max_time_list.append(max_time)
    mean_time = sum(max_time_list)/n
    for name in dict_data.keys():
        dict_data[name][1::,0] = dict_data[name][1::,0]*mean_time/(dict_data[name][-1])[0]
    for name in dict_data:
        data_list.append(dict_data[name])
    data=np.zeros([n_stamp_list,n_col])
    for j in range(n):
        n_post=n_list[j]
        n_begin=n_end
        n_end=n_begin+n_post
        for i in range(n_begin,n_end):
            data[i,:]=[try_float(x) for x in data_list[j][i-n_begin]]
    data=ordered(data)
    print("Data has been fused and normalized in time")
    print("Execution time is", mean_time ,"secondes")
    return data

# Ref: Baxter Humanoid Robot Kinematics by Robert L. Williams II
def end_pos(data):
    # Baxter geometry constants
    l0 = 270.35
    l1 = 69.00
    l2 = 364.35
    l3 = 69.00
    l4 = 374.29
    l5 = 10.00
    l6 = 368.30
    L = 278
    h = 64
    H = 1104
    n_joints=7
    #-----------------------------
    n_row, n_col = data.shape
    c = np.zeros([n_row, n_col])
    s = np.zeros([n_row, n_col])

    if n_col==n_joints:
        for i in range(n_row):
            c[i,:] = np.cos(data[i,:])
            s[i,:] = np.sin(data[i,:])

        a = s[:,0]*s[:,2] + c[:,0]*s[:,1]*c[:,2]
        b = s[:,0]*c[:,2] - c[:,0]*s[:,1]*s[:,2]
        d = c[:,0]*s[:,2] - s[:,0]*s[:,1]*c[:,2]
        f = c[:,0]*s[:,2] + s[:,0]*s[:,1]*s[:,2]
        g = s[:,1]*s[:,3] - c[:,1]*c[:,2]*c[:,3]
        h = s[:,1]*c[:,3] + c[:,1]*c[:,2]*s[:,3]

        A = a*s[:,3] - c[:,0]*c[:,1]*c[:,3]
        B = a*c[:,3] + c[:,0]*c[:,1]*c[:,3]
        D = d*s[:,3] + s[:,0]*c[:,1]*c[:,3]
        F = d*c[:,3] - s[:,0]*s[:,1]*s[:,3]
        G = g*s[:,4] - c[:,1]*s[:,2]*c[:,4]
        H = g*c[:,4] + c[:,1]*s[:,2]*s[:,4]

        x = l1*c[:,0] + l2*c[:,0]*c[:,1] - l3*a - l4*A - l5*(b*s[:,4]-B*c[:,4])
        y = l1*s[:,0] + l2*s[:,0]*c[:,1] - l3*d + l4*D + l5*(f*s[:,4]+F*c[:,4])
        z =-l2*s[:,1] - l3*c[:,1]*c[:,2] - l4*h + l5*H

        return x, y, z
    else:
        print(" Error : Number of columns doesn't correspond to number of joints")
        return None

# Compute error
def pos_error(desired_pos, real_pos):
    if desired_pos.size == real_pos.size:
        error = np.zeros(real_pos.size)
        error = desired_pos - real_pos
        return error
    else:
        print("Error : desired position and real position doesn't have the same size")

# Find the best number of components based on Bayesian Information Criterion(BIC)
def best_n_components(data,np_out=16,nc_begin=10,nc_end=50):
# BIC is computed for number of components from nc_begin to nc_end

# np_out: number of output parameters
# nc_begin : first number of components
# nc_end : last number of components
# Return the number of component equal to the lowest BIC score
    _,n_col=data.shape
    j=n_col - np_out
    if nc_begin>nc_end or np_out>n_col:
        print("Number of output y is greater than x+y or number of trials is negative")
        print("Make sure that nc_begin < nc_end and np_out< n_data_colonnes")
        return None
    else:
        bic=np.zeros(nc_end-nc_begin)
        for i in range(nc_begin,nc_end):
            gmm=GM(n_components=i,max_iter=500)
            gmm=gmm.fit(data,data[:,j::])
            bic[i-nc_begin]=gmm.bic(data)
        for i in range(bic.size):
            if bic[i]==min(bic):
                best_value=i+nc_begin
        print("Best components number is",best_value)
        return best_value

# Gaussian filter compute from a zero mean gaussian with output's shape same as input one
def gaussian_filter(data,covariance):
    n_row,n_col=data.shape
    gaus=gaussian(n_row,covariance,sym=False)
    data_fT=np.zeros([n_col,n_row])
    for i in range(n_col):
        data_fT[i,:]=np.convolve(data[:,i],gaus,mode='same')
    data_f=data_fT.T
    print("Data has been filtered")
    return data_f

# Gaussian probability compute for a data
def gaussian_dis(data,means,covariance):
# Parameters are row matrix except covariance which is a squart matrix
    d=data.size
    if d==1:
        g=(1/((np.sqrt(2*pi))*covariance)*exp(-1/2*(data-means)**2))
    else:
        v_data=data.transpose()
        v_means=means.transpose()
        vu=v_data-v_means
        vu_T=vu.transpose()
        det_cov=np.linalg.det(covariance)
        cov_I=np.linalg.inv(covariance)
        g=1/(det_cov*np.sqrt((2*pi)**d))*exp(-1/2*vu_T@cov_I@vu)
    return g

# Gaussian Mixture Regression
def regression(gmm,x):
# gmm: gaussian mixture model compute here from sklearn.mixture library
# x: variable from y=f(x) where y is the regression output
# Return: regression
    n=gmm.n_components
    weights=gmm.weights_
    means=gmm.means_
    cov=gmm.covariances_
    if x.shape==((x.shape)[0],):
        x_col=1
        x_row=(x.shape)[0]
    else :
        [x_row,x_col]=x.shape
    n_means=(means[0]).size
    n_out_param=n_means-x_col
    reg=np.zeros([x_row,n_out_param])
    sum_g=np.zeros([x_row,1])
    x_prob=np.zeros([x_row,n])
    h_k=np.zeros([x_row,n])
    buf=np.zeros([n_out_param,1])

    # calculation normal(x,means,covariances) for each point point and for each cluster
    if x_col==1 :
        x_reg=np.zeros([x_row,n_out_param+1])
        for j in range(x_row):
            for i in range(n):
                x_prob[j,i]=gaussian_dis(x[j],means[i,0],cov[i,0,0])
        # calculation of sum of weight*normal(x,means,covariances) for each point
        for j in range(x_row):
            for i in range(n):
                sum_g[j]=sum_g[j]+weights[i]*x_prob[j,i]
        # calculation of h(x) for each cluster and each point
        for j in range(x_row):
            for i in range(n):
                h_k[j,i]=weights[i]*x_prob[j,i]/sum_g[j]
        for j in range(x_row):
            for i in range(n):
                inv_covx=1/cov[i,0,0]
                buf=means[i,x_col::]+(cov[i,x_col::,0:x_col]*inv_covx*(x[j]-means[i,0]).T).T
                reg[j,:]=reg[j,:]+h_k[j,i]*buf
        x_reg[:,0]=x
        x_reg[:,1::]=reg
    else:
        for j in range(x_row):
            for i in range(n):
                x_prob[j,i]=gaussian_dis(x[j,:],means[i,0:x_col],cov[i,0:x_col,0:x_col])
        # calculation of sum of weight*normal(x,means,covariances) for each point
        for j in range(x_row):
            for i in range(n):
                sum_g[j]=sum_g[j]+weights[i]*x_prob[j,i]
        # calculation of h(x) for each cluster and each point
        for j in range(x_row):
            for i in range(n):
                h_k[j,i]=weights[i]*x_prob[j,i]/sum_g[j]
        for j in range(x_row):
            for i in range(n):
                inv_covx=np.linalg.inv(cov[i,0:x_col,0:x_col])
                buf=means[i,x_col::].T+cov[i,x_col::,0:x_col]@inv_covx@(x[j,:]-means[i,0:x_col]).T
                reg[j,:]=reg[j,:]+h_k[j,i]*buf.T
        x_reg=np.concatenate((x,reg), axis=1)
    return x_reg
