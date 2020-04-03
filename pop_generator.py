''' Source code written by Yule Vaz '''

#
# @package popmob
#

import numpy as np

# Generate adjacency matrix with ones, for no-directional graphs
# @param N	Number of nodes
def gen_adj_ones(N):

	arr = np.ones((N,N),dtype=float)
	np.fill_diagonal(arr,0)
 
	return arr
		
# Generate adjacency matrix with binomial distribution, for no-directional graphs
# @param N	Number of nodes
# @param p	Probability of ones
def gen_adj_unif(p,N):

	arr = np.zeros((N,N),dtype=float)
	for i in range(1,N):
		arr[i-1,i:N] = np.random.binomial(1,p,N-i)
		arr[i:N,i-1] = arr[i-1,i:N]
 
	return arr.reshape((N,N))
		
 

# Generate population by Chi-square distribution.
# @param min_pop	Minimum population considered
# @param max_pop	Maximum population considered
# @param df 		Degree of freedom of Chi-square distribution
# @param N		Number of regions
# @return numpy.array 	A array with populations size
def gen_pop_chi2(min_pop,max_pop,df,N):

	chi = np.random.chisquare(df,N)
	P = (chi - np.min(chi))/(np.max(chi)-np.min(chi))
	dpop = max_pop - min_pop
	
	return np.floor(min_pop + dpop * P)
