import getopt,sys
import networkx as nx
import numpy as np
from numpy import linalg
from scipy.stats import pearsonr
import math
import glob

def mnorm(array):
	return (array - np.min(array)) / (np.max(array) - np.min(array))

class EllipsoidTool:
    """This particular class code is taken from https://github.com/minillinim/ellipsoid"""
    def __init__(self): pass
    
    def getMinVolEllipse(self, P=None, tolerance=0.01):
        (N, d) = np.shape(P)
        d = float(d)
    
        # Q will be our working array
        Q = np.vstack([np.copy(P.T), np.ones(N)]) 
        QT = Q.T
        
        # initializations
        err = 1.0 + tolerance
        u = (1.0 / N) * np.ones(N)

        # Khachiyan Algorithm
        while err > tolerance:
            V = np.dot(Q, np.dot(np.diag(u), QT))
            M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
            j = np.argmax(M)
            maximum = M[j]
            step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
            new_u = (1.0 - step_size) * u
            new_u[j] += step_size
            err = np.linalg.norm(new_u - u)
            u = new_u

        # center of the ellipse 
        center = np.dot(P.T, u)
    
        # the A matrix for the ellipse
        A = linalg.inv(
                       np.dot(P.T, np.dot(np.diag(u), P)) - 
                       np.array([[a * b for b in center] for a in center])
                       ) / d
                       
        # Get the values we'd like to return
        U, s, rotation = linalg.svd(A)
        radii = 1.0/np.sqrt(s)
        
        return (center, radii, rotation)

options, reminder = getopt.getopt(sys.argv[1:],'l:n:g:s:',['label','samples','nxfile','source'])

for opt, arg in options:
    if opt in ('-l','--label'):
        labelfile = arg
    elif opt in ('-n','--samples'):
        num_samples = int(arg)
    elif opt in ('-g','--nxfile'):
        nwxfile = arg
    elif opt in ('-s','--source'):
        sd = arg

node_label = np.loadtxt(labelfile,dtype='int32')

chrm = [_ for _ in range(1,23)] + ['X']

samples = num_samples

G = nx.read_gexf(nwxfile)

GM = max(nx.connected_component_subgraphs(G), key=len)

cnodes = GM.nodes()

chrcounts = np.zeros(len(chrm))

distArray = np.zeros((samples,len(chrm)))

gene_density = np.array([7.86,4.87,5.20,3.77,4.62,5.86,5.37,4.36,5.30,5.27,9.16,7.37,2.65,5.37,5.33,8.67,13.68,3.29,22.53,8.22,4.43,8.15,5.19])
chr_len = np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540, \
			102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chr_len = chr_len / 1E6

gene_density = mnorm(gene_density)

chr_len = mnorm(chr_len)

file_list = [sd+'/csn_'+str(_)+'_coor.txt' for _ in range(1,samples+1)]

for findex,file in enumerate(file_list):
    print(findex)
    G_coor = np.loadtxt(file)
    chrcoords = [[] for _ in chrm]
    chrcenter = np.zeros((len(chrm),3))
    for i in range(len(node_label)):
        if str(i) in cnodes:
            chrcounts[node_label[i]] += 1
            chrcenter[node_label[i]] += G_coor[i,:]
    for i in range(len(chrm)):
        chrcenter[i] = chrcenter[i] / chrcounts[i]
    ET = EllipsoidTool()
    (system_center, radii, rotation) = ET.getMinVolEllipse(G_coor, .01)
    chr_dist = []
    chr_odist = np.zeros(len(chrm))
    for i in range(23):
        chr_odist[i] = np.sqrt(np.sum((chrcenter[i]-system_center) ** 2))
    distArray[findex] = mnorm(chr_odist)

np.savetxt(sd+'Dist.txt',distArray)


