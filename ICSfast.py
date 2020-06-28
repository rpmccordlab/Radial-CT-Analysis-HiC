"""Initial Centroid Selection For MK-means Clustering Algorithm"""
__author__ = 'Priyojit Das'

import numpy as np
from math import sqrt
import math
import sklearn
import scipy.stats as sp
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

def axis_selection(data):
    """Feature Selection Technique"""
    sd = np.std(data,axis=0)
    mean = np.mean(data,axis=0)
    x = np.argmax(np.abs(sd/mean))
    cmat = np.corrcoef(data.T)
    y = np.argmin(cmat[x,:])
    return data[:,[x,y]]

def dist_func(c1,c2,dist):
    """ Distance Calculation Function"""
    if dist == "euclid":
        return sqrt(sum((c1 - c2) ** 2))
    if dist == "pearson":
        """r = sp.pearsonr(c1,c2)[0]
        if r < 0:
            return 1.0
        else:
            return (1.0 - r)"""
        #print sp.pearsonr(c1,c2)
        return (1 - sp.pearsonr(c1,c2)[0])

def dist_mat_const(data,dist):
    """Distance Matrix Calculation"""
    n = data.shape[0]
    narr = euclidean_distances(data)
    return narr

def find_minimun(data,nodes):
    """Find The Pair of Point Having Minimum Distance Between Them"""
    fdist = deepcopy(data[nodes,:])
    fdist = fdist[:,nodes]
    np.fill_diagonal(fdist, np.max(fdist)+1)
    fmin = np.argmin(fdist)
    a,b = int(np.divide(fmin,len(nodes))),np.mod(fmin,len(nodes))
    return [nodes[a],nodes[b]]

def find_nearest(x,nodes,data):
    """Find The Nearest Point With Respect To Another Point"""
    dist = (data[nodes,:] - x)**2
    dist = np.sum(dist, axis=1)
    dist = np.sqrt(dist)
    return nodes[np.argmin(dist)]

def find_farthest(x,nodes,data):
    """Find The Farthest Point With Respect To Another Point"""
    dist = []
    for i in nodes:
        dist.append(dist_func(x,data[i,:],"euclid"))
    return nodes[np.argmax(dist)]

def quantize_cluster(C,data):
    """Select Centroid of Each Cluster"""
    qC = []
    for i in C:
        mC = np.mean(data[i,:],axis=0)
        pos = find_nearest(mC,i,data)
        qC.append(pos)
    return qC

def find_cluster_centroid(data,K):
    """Identify Cluster Centroids"""
    #if data.shape[1] > 2:
    #    data = axis_selection(data)
    n = data.shape[0]
    cV = int(0.75 * math.ceil(n/float(K)))
    Cluster = [[] for i in range(K)]
    all_nodes = [i for i in range(n)]
    #sim_mat = sklearn.metrics.pairwise.euclidean_distances(data)
    sim_mat = dist_mat_const(data,"euclid")
    minpos = find_minimun(sim_mat,all_nodes)
    i,j = minpos[0],minpos[1]
    #print j,Cluster
    #print i,j
    Cluster[0].append(i)
    Cluster[0].append(j)
    del all_nodes[all_nodes.index(i)]
    del all_nodes[all_nodes.index(j)]
    for i in range(K):
        #print Cluster,cV
        if i > 0:
            minpos = find_minimun(sim_mat,all_nodes)
            a,b = minpos[0],minpos[1]
            #print j,Cluster
            #print i,j
            Cluster[i].append(a)
            Cluster[i].append(b)
            del all_nodes[all_nodes.index(a)]
            del all_nodes[all_nodes.index(b)]
        while True:
            if len(Cluster[i]) == cV:
                break
            temp = sim_mat[Cluster[i],:]
            temp = temp[:,all_nodes]
            imax = np.argmin(np.min(temp,axis=0))
            Cluster[i].append(all_nodes[imax])
            del all_nodes[imax]
    return quantize_cluster(Cluster,data)



