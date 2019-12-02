import getopt,sys
import networkx as nx
import numpy as np
from numpy import linalg
from scipy.stats import pearsonr, spearmanr
import math
import glob
from sklearn.svm import SVR
from loess.loess_1d import loess_1d


def znorm(array):
	return (array - np.mean(array)) / np.std(array)

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

options, reminder = getopt.getopt(sys.argv[1:],'l:n:g:t:s:',['label','samples','nxfile','ntype','source'])

for opt, arg in options:
    if opt in ('-l','--label'):
        labelfile = arg
    elif opt in ('-n','--samples'):
        num_samples = int(arg)
    elif opt in ('-g','--nxfile'):
        nwxfile = arg
    elif opt in ('-t','--ntype'):
        cntype = arg
    elif opt in ('-s','--source'):
        sd = arg

node_label = np.loadtxt(labelfile,dtype='int32')

chrm = [_ for _ in xrange(1,23)] + ['X']

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

z = loess_1d(gene_density,chr_len,frac=2./3)

file_list = [sd+'/csn_'+str(_)+'_coor.txt' for _ in xrange(1,samples+1)]

for findex,file in enumerate(file_list):
    #print findex
    G_coor = np.loadtxt(file)
    chrcoords = [[] for _ in chrm]
    chrcenter = np.zeros((len(chrm),3))
    for i in xrange(len(node_label)):
        if str(i) in cnodes:
            chrcounts[node_label[i]] += 1
            chrcenter[node_label[i]] += G_coor[i,:]
    for i in xrange(len(chrm)):
        chrcenter[i] = chrcenter[i] / chrcounts[i]
    ET = EllipsoidTool()
    (system_center, radii, rotation) = ET.getMinVolEllipse(G_coor, .01)
    chr_dist = []
    chr_odist = np.zeros(len(chrm))
    for i in xrange(23):
        chr_odist[i] = np.sqrt(np.sum((chrcenter[i]-system_center) ** 2))
    distArray[findex] = mnorm(chr_odist)


if cntype == 'L':
    chrN = np.array([19,22,17,16,1,15,12,21,14,6,20,9,10,2,5,11,3,8,18,7,23,13,4]) # Boyle
    chrNpos = np.array([0.229,0.249,0.275,0.290,0.299,0.330,0.330,0.339,0.339,0.339,0.349,0.349,0.349,0.349,0.360,0.360,0.371,0.371,0.371,0.371,0.371,0.371,0.389]) # Boyle
    X = 0.1 * mnorm(z[1]) + 0.6 * mnorm(np.exp(1 - gene_density)) + 0.3 * chr_len
elif cntype == 'F':
    chrN = np.array([22,21,18,17,20,14,16,15,19,11,12,13,6,10,9,8,7,1,23,3,2,4,5]) # Bolzer
    chrNpos = np.array([0.53,0.55,0.57,0.58,0.61,0.62,0.62,0.63,0.63,0.65,0.65,0.66,0.68,0.68,0.72,0.73,0.74,0.75,0.75,0.76,0.77,0.77,0.79]) # Bolzer
    X = 0.1 * mnorm(z[1]) + 0.3 * mnorm(np.exp(1 - gene_density)) + 0.6 * chr_len


dist_fil = np.array(distArray).T

mreg = SVR(kernel='linear').fit(dist_fil,X)
predval = mreg.predict(dist_fil)

if cntype == 'L':
    print "Predicted CT distance from Nucleus Center with Spherical Fit:"
elif cntype == 'F':
    print "Predicted CT distance from Nucleus Center with Ellipsoidal Fit:"
print predval
print 
if cntype == 'L':
    print "Correlation with the Lymphoblastoid Microscopy Imaging Data:", pearsonr(predval[chrN-1],chrNpos)[0]
elif cntype == 'F':
    print "Correlation with the Fibroblast Microscopy Imaging Data:", pearsonr(predval[chrN-1],chrNpos)[0]


