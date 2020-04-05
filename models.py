''' Source code written by Yule Vaz'''

#
# @package popmob
#

import numpy as np
import scipy as sci


# @param b			The beta parameter
# @param c			The gamma parameter
# @param S			Previous conditions S
# @param I			Previous condition I
# @param R			Previous conditions R
# @return float,float,float 	Stock, infected and recovery values
def SIR_model(b,c,S,I,R):
	
	Sf = S - (b *  I * S)/(S + I + R)
	If = I + (b * I * S)/(S + I + R) - c * I
	Rf = R + c * I
 
	return Sf,If,Rf 

# Arising of disease in other region
# @param probs	Probability of a disease to appear
# @return numpy.array A binary vector associated with the existence of the disease (1 for exist, 0 for don't exist) 
def appearance(probs):

	probs = np.array(probs)
	appear = np.array(len(probs)*[1])
	rands = np.random.uniform(0,1,len(probs))
	idx = np.where(probs-rands < 0)[0]
	if len(idx) > 0:
		appear[idx] = 0
	
	return appear

# Create SIR-yan model verifying the lockdown for regions A = [a1,a2,...,an] in time t = [t1,t2,...,tn]
# @param b		Beta param for SIR
# @param c		Gamma param for SIR
# @param pops		Population vector
# @param P0	 	The Yan transition probabilities
# @param days		Number of days
# @param quar		Regions of quarentine
# @param when_quar	When to quarentine
# @param lock		Lockdown
# @param when_lock	When to lockdown
# @param first		List with the regions where disease first appears
# @param a_lock		Percentage of lockdown 
def yan_SIR(b,c,pops,P0,days,quar,when_quar,lock,when_lock,first,a_lock):

	L = len(pops)
	S = np.array(pops).reshape((1,L))
	I = np.zeros((1,L),dtype=float)
	I[0,first] = 1
	R = np.zeros((1,L),dtype=float)
	P = P0
	T = []
	step = 0

	# Cast to numpy.array in order to allow indexing with list
	when_quar = np.array(when_quar)
	when_lock = np.array(when_lock)
	quar = np.array(quar)
	lock = np.array(lock)
	b = np.array(b)
	c = np.array(c)
	# Factor of dispersion of disease are fixed yet...

	for ti in range(0,days):
		Sr,Ir,Rr = SIR_model(b,c,S[step,:],I[step,:],R[step,:])
		
		# Quarentine
		idx = np.array(np.where(when_quar == step)[0])
		qdi = quar[idx]
		P[qdi.tolist(),:] = 0
		P[:,qdi.tolist()] = 0
		# Lockdown
		idx = np.array(np.where(when_lock <= step)[0])
		ldi = lock[idx]
		b[ldi] = b[ldi]*(1-a_lock)
		P[ldi.tolist(),:] = 0
		P[:,ldi.tolist()] = 0

		# Mobility	
		Ps = np.matmul(P,Sr)
		Pi = np.matmul(P,Ir)
		transf = b * Sr * Pi /(pops + np.sum(P)) 
		Sr = Sr - transf
		
		idx1 = np.where(Ir < 1)
		m = np.piecewise(Ir,[Ir < 1, Ir >= 1],[0,1])
		m[idx1] = appearance(Pi[idx1]) 
		Ir = Ir + m*transf
		T.append(np.sum(Ir))	

		# Store values
		S = np.vstack((S,Sr))
		R = np.vstack((R,Rr))
		I = np.vstack((I,Ir))	

		# Update mobility
		P = np.matmul(P0,P)

		step += 1
			
		
	return S,I,R,T

# Create population mobilization model based in "Universal model of individual and population mobility on diverse spatial scales" (Yan, X. Y.; et. al; 2017) (https://www.nature.com/articles/s41467-017-01892-8#ref-CR6). However, the number of visits regarded in (Yan, X. Y.; et. al.) is simplified by initially ranking $m_i / w_{a,i}$.
# @param pops		Populations of each regions
# @param mem		Memory constant 
# @param A		Adjacency matrix
# @param tol		Tolerance for adjacency matrix, insert "tol" instead 0
# @return numpy.array	Probability transition matrix
def yan_etal_model(pops,mem,A,tol=0):

	B = A
	W = np.matmul(A,pops)
	mf = np.divide(pops,np.sum(W,axis=0))
	P = np.zeros((len(mf),len(mf)),dtype=float)

	#sort ranks for m_i
	r = mf.argsort()[::-1].argsort() + 1

	B[B<1] = tol 

	for i in range(0,P.shape[0]):

		for j in range(0,P.shape[1]):
		
			factor = 1 + mem/r[j]
			P[i,j] = mf[j] * factor * A[i,j]

		P[i,:] = P[i,:] / np.sum(P[i,:])
	
	return np.transpose(P)
