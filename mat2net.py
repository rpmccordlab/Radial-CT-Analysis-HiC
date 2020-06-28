import numpy as np
import networkx as nx
import pandas as pd
import pickle
import random
import sys,getopt
import os

def input_nd_process_hic(loc):
	df = pd.read_csv(loc,comment='#',sep='\t')
	df = df.set_index(list(df)[0])
	df = df.dropna(axis='columns',how='all')
	df = df.dropna(axis='index',how='all')	
	df = df.fillna(0)
	indx = list(df)
	indx = [_.split("|")[2] for _ in indx]
	return df.values, indx

random.seed(1000)

options, reminder = getopt.getopt(sys.argv[1:],'i:n:o:',['input','samples','outdir'])

for opt, arg in options:
	if opt in ('-i','--input'):
		filename = arg
	elif opt in ('-n','--samples'):
		num_samples = int(arg)
	elif opt in ('-o','--outdir'):
		out = arg

mat,indx = input_nd_process_hic(filename)

if not os.path.isdir(out):
	os.system("mkdir "+out)

dloc = []

for it,_ in enumerate(indx):
    chrID, loc = _.split(":")
    if chrID in ['chrY','chrM']:
        dloc.append(it)

mat = np.delete(mat,dloc,0)
mat = np.delete(mat,dloc,1)
indx = np.delete(indx,dloc)

cutoff = np.percentile(mat[np.triu_indices(mat.shape[0])],95)

mat = mat * (mat > cutoff)

chrm = [_ for _ in range(1,23)] + ['X']

residue_number = len(indx)

node_label = []

for it,_ in enumerate(indx):
    chrID, loc = _.split(":")
    if chrID in ['chrX']:
    	chrID = 22
    else:
    	chrID = int(chrID.split('chr')[-1])-1
    node_label.append(chrID)

np.savetxt(filename+'.label',node_label)

G = nx.Graph()

for i in range(residue_number):
	G.add_node(i)

for i in range(mat.shape[0]-1):
	for j in range(i+1,mat.shape[0]):
		if mat[i,j]:
			G.add_edge(i,j,weight=mat[i,j])

for rindex,rint in enumerate(range(1,num_samples+1)):
	#print rindex
	pos = nx.spring_layout(G,dim=3,seed=rint,iterations=500)
	xyz = [list(pos[i]) for i in pos]
	np.savetxt(out+'/csn_'+str(rint)+'_coor.txt',xyz)

nx.write_gexf(G,filename+'.gexf',version='1.2draft')
