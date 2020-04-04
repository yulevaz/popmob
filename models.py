''' Source code written by Yule Vaz'''

#
# @package popmob
#

import numpy as np
import scipy as sci

# Create SIR model for epidemic prediction following eq. 4 of pg. 2 in the pre-print "Exact Solution to a Dynamic SIR Model"(Bohner, M.; Streipert, S.; Torres, D. F. M.; 2018).
# @param b			The beta parameter
# @param c			The gamma parameter
# @param S0			Initial conditions S
# @param I0			Initial conditions I
# @param B			Initial conditions N
# @param time			Time
# @return float,float,float 	Stock, infected and recovery values
def SIR_model(b,c,S0,I0,N,t):

	k = I0/S0
	S = S0 * ((1+k)**(b/(b-c)))*((1+k*np.exp((b-c)*t))**(-b/(b-c)))
	I = I0 * ((1+k)**(b/(b-c)))*((1+k*np.exp((b-c)*t))**(-b/(b-c)))*np.exp((b-c)*t)
	R = N - S - I
 
	return S,I,R 

# Arising of disease in other region
# @param probs	Probability of a disease to appear
# @return numpy.array A binary vector associated with the existence of the disease (1 for exist, 0 for don't exist) 
def appearance(probs):

	probs = np.array(probs)/(np.sum(probs)+1e-10)
	appear = np.array(len(probs)*[1])
	rands = np.random.uniform(0,1,len(probs))
	idx = np.where(probs-rands < 0)[0]
	if len(idx) > 0:
		appear[idx] = 0
	
	return appear
	

# Create SIR-yan model verifying the lockdown for regions A = [a1,a2,...,an] in time t = [t1,t2,...,tn]
# @param pops		Population vector
# @param P0	 	The Yan transition probabilities
# @param t		Time vector
# @param epi_model 	Lambda function with epidemic model (lambda N,t : SIR_model(b,c,S0,I0,N,t))
# @param quar		Regions of quarentine
# @param when_quar	When to quarentine
# @param lock		Lockdown
# @param when_lock	When to lockdown 
# @param first		List with the regions where disease first appears
def yan_epidemic(pops,P0,t,epi_model,quar,when_quar,lock,when_lock,first):

	L = len(pops)
	S = np.array(pops)
	I = np.zeros((1,L),dtype=float)
	I[0,first] = 1
	R = np.zeros((1,L),dtype=float)
	S0 = S
	I0 = I[0,:]
	P = P0
	T = []
	tr = np.array(L*[0])
	step = 0
	when_quar = np.array(when_quar)
	when_lock = np.array(when_lock)
	quar = np.array(quar)
	lock = np.array(lock)
	# Factor of dispersion of disease are fixed yet...
	#TODO: Factorize
	b = np.array(L * [3.0])
	c = np.array(L * [1.0])

	for ti in t:
		mp = map(epi_model,b,c,S0,I0,pops,t[tr])
		npL = np.array(list(mp))
		S = np.vstack((S,npL[:,0]))
		R = np.vstack((R,npL[:,2]))
		# Quarentine
		idx = np.array(np.where(when_quar == step)[0])
		qdi = quar[idx]
		P[qdi.tolist(),:] = 0
		P[:,qdi.tolist()] = 0
		# Lockdown
		idx = np.array(np.where(when_lock == step)[0])
		ldi = lock[idx]
		if len(ldi) > 0:
			b[ldi.tolist()] = 1.5
			P[ldi.tolist(),:] = 0
			P[:,ldi.tolist()] = 0
			

		I = np.vstack((I,npL[:,1]))
		# Mobilization of infected
		Iax = np.matmul(P,I[step+1,:])
		id1 = np.where(I0 >= 1)[0]
		Iax[id1] = I0[id1]
		I0 = Iax
		nid1 = np.delete(range(0,L),id1)
		I0[nid1] = appearance(I0[nid1])
		S0 = S[0,:] - I0
		tr[id1] += 1
		# Mobilization of population
		pops = np.matmul(P,pops)
		P = np.matmul(P0,P)
		T.append(np.sum(I[step+1,:]))
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
