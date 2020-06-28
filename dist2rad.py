import getopt,sys
import numpy as np
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from loess.loess_1d import loess_1d
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib
from kneed import KneeLocator
from scipy.spatial.distance import cdist
from sklearn.metrics import davies_bouldin_score,calinski_harabasz_score,silhouette_score

from copy import deepcopy
from ICSfast import *

np.random.seed(1000)

def mnorm(array):
	return (array - np.min(array)) / (np.max(array) - np.min(array))

def estimate_k(dataK):
	ks = range(2, 31)
	inertias = []
	sils = []
	for k in ks:
		#print(k)
		centroids = find_cluster_centroid(dataK,k)
		# Create a KMeans instance with k clusters: model
		model = KMeans(n_clusters=k,init=dataK[centroids,:])
		# Fit model to samples
		model.fit(dataK)
		inertias.append(model.inertia_)
		sils.append(silhouette_score(dataK,model.labels_))
   
	kn = KneeLocator(ks, inertias, curve='convex', direction='decreasing', interp_method='polynomial')
	spos = ks.index(kn.knee)
	spos = sils.index(np.max(sils[spos-2:spos+3]))
	#print(kn.knee,ks[spos])
	return ks[spos]

def input_nd_process_hic(loc):
    df = pd.read_csv(loc,sep='\t',comment='#')
    df = df.set_index(list(df)[0])
    df = df.dropna(axis='columns',how='all')
    df = df.dropna(axis='index',how='all')	
    df = df.fillna(0)
    indx = list(df)
    indx = [_.split("|")[2] for _ in indx]
    dloc = []
    for it,_ in enumerate(indx):
    	chrID, loc = _.split(":")
    	if chrID in ['chrY','chrM']:
    		dloc.append(it)
    mat = df.values
    mat = np.delete(mat,dloc,0)
    mat = np.delete(mat,dloc,1)
    indx = np.delete(indx,dloc)
    return mat, indx

def trans_pattern(loc):
	mat, indx = input_nd_process_hic(loc)
	intrc = np.zeros((23,23),dtype='int32')
	locdic = {}

	for _ in range(25):
		locdic[_] = 0

	for _ in indx:
		val =  _.split(':')[0][3:]
		if val == 'X':
			val = 23
		elif val == 'Y':
			val = 24
		elif val == 'M':
			val = 25
		val = int(val)-1
		locdic[val] += 1

	cutoff = np.percentile(mat[np.triu_indices(mat.shape[0])],95)

	mask = np.array(mat > cutoff,dtype='int32')

	dloc = [0]

	for _ in range(25):
		dloc.append(dloc[-1]+locdic[_])

	for i in range(23):
		for j in range(i,23):
			intrc[i,j] = intrc[j,i] = np.sum(mask[dloc[i]:dloc[i+1],dloc[j]:dloc[j+1]])

	uval = []

	for i in range(23):
		for j in range(i+1,23):
			uval.append(intrc[i,j])

	umin = np.min(uval)
	umax = np.max(uval)

	upmat = np.zeros((23,23))

	for i in range(23):
		for j in range(i+1,23):
			#upmat[j,i] = upmat[i,j] = (intrc[i,j] - umin) / float(umax - umin)
			upmat[j,i] = upmat[i,j] = (intrc[i,j]) / float(umax)
	
	return upmat

def estimate_order(file):
	hdata = trans_pattern(file)
	pca = PCA(n_components=2)
	pc = pca.fit_transform(hdata)
	gcorr = pearsonr(pc[:,0],gene_density)[0]
	lcorr = pearsonr(pc[:,0],chr_len)[0]
	if np.abs(gcorr) > np.abs(lcorr):
		return "S"
	else:
		return "E"

def cp_tune(dataT):
	dataCorr = np.corrcoef(dataT.T)
	pca = PCA(n_components=2)
	pc = pca.fit_transform(dataCorr)
	pervals = pca.explained_variance_ratio_[0:2]
	pervals = pervals / np.sum(pervals)
	MX = pervals[1] * pc[:,0] + pervals[0] * np.mean(dataT,axis=0)
	SX = pervals[0] * mnorm(np.exp(1 - gene_density)) + pervals[1] * chr_len
	EX = pervals[1] * mnorm(np.exp(1 - gene_density)) + pervals[0] * chr_len
	GX = loess_1d(SX,MX,frac=2/3.0)[1]
	LX = loess_1d(EX,MX,frac=2/3.0)[1]
	return (GX,LX)


gene_density = np.array([7.86,4.87,5.20,3.77,4.62,5.86,5.37,4.36,5.30,5.27,9.16,7.37,2.65,5.37,5.33,8.67,13.68,3.29,22.53,8.22,4.43,8.15,5.19])
chr_len = np.array([249250621,243199373,198022430,191154276,180915260,171115067,159138663,146364022,141213431,135534747,135006516,133851895,115169878,107349540, \
			102531392,90354753,81195210,78077248,59128983,63025520,48129895,51304566,155270560])

chrNL = np.array([19,22,17,16,1,15,12,21,14,6,20,9,10,2,5,11,3,8,18,7,23,13,4]) # Boyle
chrNLpos = np.array([0.229,0.249,0.275,0.290,0.299,0.330,0.330,0.339,0.339,0.339,0.349,0.349,0.349,0.349,0.360,0.360,0.371,0.371,0.371,0.371,0.371,0.371,0.389]) # Boyle
chrNF = np.array([22,21,18,17,20,14,16,15,19,11,12,13,6,10,9,8,7,1,23,3,2,4,5])
chrNFpos = np.array([0.53,0.55,0.57,0.58,0.61,0.62,0.62,0.63,0.63,0.65,0.65,0.66,0.68,0.68,0.72,0.73,0.74,0.75,0.75,0.76,0.77,0.77,0.79])
chrNM = np.array([21,16,18,15,12,11,4,1]) # Firtz
chrNMpos = np.array([0.482,0.549,0.564,0.593,0.614,0.659,0.682,0.736]) # Firtz

chr_len = chr_len / 1E6

gene_density = mnorm(gene_density)

chr_len = mnorm(chr_len)

options, reminder = getopt.getopt(sys.argv[1:],'i:d:',['input','distance'])

for opt, arg in options:
    if opt in ('-i','--input'):
        hfilename = arg
    elif opt in ('-d','--distance'):
        dfilename = arg

order_t = estimate_order(hfilename) 

data = np.loadtxt(dfilename)

k = estimate_k(data)

print(hfilename)

centroids = find_cluster_centroid(data,k)

model = KMeans(n_clusters=k,init=data[centroids,:])
# Fit model to samples
model.fit(data)

labels = model.labels_ + 1

gd_mean_arr = []
ln_mean_arr = []

for i in range(1,k+1):
	flag = (labels == i)
	data_c = data[flag,:]
	gd_mean_arr.append(pearsonr(np.mean(data_c,axis=0),gene_density)[0])
	ln_mean_arr.append(pearsonr(np.mean(data_c,axis=0),chr_len)[0])

if order_t == 'S':
	gsign = ""
	gmvp = np.argsort(gd_mean_arr)
	if np.abs(gd_mean_arr[gmvp[0]]) >= np.abs(gd_mean_arr[gmvp[-1]]):
		gd_idx = gmvp[0] + 1
		if gd_mean_arr[gmvp[0]] < 0:
			gsign = "+"
		else:
			gsign = "-"
	else:
		gd_idx = gmvp[-1] + 1
		if gd_mean_arr[gmvp[-1]] < 0:
			gsign = "+"
		else:
			gsign = "-"

	data_gd = data[labels == gd_idx,:]

	if gsign == '-':
		data_gd = 1 - data_gd

	gd_order = cp_tune(data_gd)
	predval = mnorm(gd_order[0]) 
else:
	lsign = ""
	lmvp = np.argsort(ln_mean_arr)
	if np.abs(ln_mean_arr[lmvp[0]]) >= np.abs(ln_mean_arr[lmvp[-1]]):
		ln_idx = lmvp[0] + 1
		if ln_mean_arr[lmvp[0]] < 0:
			lsign = "-"
		else:
			lsign = "+"
	else:
		ln_idx = lmvp[-1] + 1
		if ln_mean_arr[lmvp[-1]] < 0:
			lsign = "-"
		else:
			lsign = "+"

	data_ln = data[labels == ln_idx,:]

	if lsign == '-':
		data_ln = 1 - data_ln

	ln_order = cp_tune(data_ln)
	predval = mnorm(ln_order[1]) 

if order_t == 'S':
    print("Inferred CT Distribution Type: Gene Density Driven")
elif order_t == 'E':
    print("Inferred CT Distribution Type: Chromosome Legth Driven")
print("Predicted CT distance from Nucleus Center:")
print(mnorm(predval))
print() 
if order_t == 'S':
    print("Correlation with the Lymphoblastoid Microscopy Imaging Data:", pearsonr(predval[chrNL-1],chrNLpos)[0])
elif order_t == 'E':
    print("Correlation with the Fibroblast Microscopy Imaging Data:", pearsonr(predval[chrNF-1],chrNFpos)[0])