''' Source code written by Yule Vaz'''

#
# @package popmob
#

import numpy as np

# Create population mobilization model based in "Universal model of individual and population mobility on diverse spatial scales" (Yan, X. Y.; et. al.) (https://www.nature.com/articles/s41467-017-01892-8#ref-CR6). However, the number of visits regarded in (Yan, X. Y.; et. al.) is simplified by initially ranking $m_i / w_{a,i}$.
# @param pops	Populations of each regions
# @param mem	Memory constant 
# @param A	Adjacency matrix
# @param tol	Tolerance for adjacency matrix, insert "tol" instead 0
def yan_etal_model(pops,mem,A,tol=0):

	B = A
	W = np.matmul(A,pops)
	mf = np.divide(pops,np.sum(W,axis=0))
	P = np.zeros((len(mf),len(mf)),dtype=float)

	#sort ranks for m_i
	r = mf.argsort()[::-1].argsort() + 1

	B[B<1] = tol 

	print(A)
	for i in range(0,P.shape[0]):

		for j in range(0,P.shape[1]):
		
			factor = 1 + mem/r[j]
			P[i,j] = mf[j] * factor * A[i,j]

		P[i,:] = P[i,:] / np.sum(P[i,:])
	
	return P		
