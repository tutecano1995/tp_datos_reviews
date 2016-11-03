# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 01:39:51 2016

@author: joaquintz
"""
import base64
import csv
import numpy as np
import json
import matplotlib as mpl
mpl.use('TkAgg')
import pylab as Plot



#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

def Hbeta(D = np.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P) + 1e-12;
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0,metric="euclid"):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
 	if metric == "euclid":
		sum_X = np.sum(np.square(X), 1);
		D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
 	elif metric == "cosine":
		phi = np.sqrt(np.sum(np.square(X),1))
		D = np.divide(X.T,phi).T #normalize all rows
		del phi
		D = 1- np.dot(D,D.T)# cosine distance
		D[range(n),range(n)] = 0
	
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf;
		betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
	return P;


def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0, metric ="euclid"):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = np.random.randn(n, no_dims);
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity,metric);
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;									# early exaggeration
	P = np.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1);
		num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / np.sum(num);
		Q = np.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - np.tile(np.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = np.sum(P * np.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict 
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64,
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


#dim_vec = 200

#with open("/home/joaquintz/Desktop/facultad/datos/reviews_processed.csv") as f:
#    reader = csv.reader(f)
#    reader.next() # saltamos el header
#    reviews = list(reader)    

#matriz = np.zeros((len(reviews),dim_vec))

#i=0
#for review in reviews:
#    sentence = review[10]
#    words = sentence.split(' ')
#    for word in words:
#        if word == '': continue
#        j=hash(word) % dim_vec
#        matriz[i][j] += 1
#    i+=1


#elegimos 10k filas al azar
def elegir_filas_random(reviews):
    return np.random.randint(1,len(reviews),8000)

def elegir_filas_estrellas_homogeneas(reviews):
    contador_estrellas=np.array([0]*5)
    cant_reviews=len(reviews)
    filas=[]
    total=0
    
    while total<1500*5:
        indice = np.random.randint(1,cant_reviews)
        estrellas_indice = int(reviews[indice][7])
        if contador_estrellas[estrellas_indice-1]<1500:
            contador_estrellas[estrellas_indice-1]+=1
            filas.append(indice)
            total+=1
    return filas

#filas = elegir_filas_random(reviews) 
#filas=elegir_filas_estrellas_homogeneas(reviews)
#estrellas = [int(reviews[k][7]) for k in filas]
#matriz_red = matriz[filas]

#attenti: el codigo de arriba puede agarrar algunas filas dobles..
#colourmap= {1:'#F64646',2:'#E17133',3:'#CC9C22',4:'#86AD0E',5:'#1F8F00'}
#colours = [colourmap[n] for n in estrellas]

#reduced_set =tsne(matriz_red,no_dims=2,initial_dims=20,perplexity=20.0,metric="cosine")

#star_hist=np.array([0]*5)
#for j in estrellas:
#    star_hist[j-1]+=1

#Plot.scatter(reduced_set[:,0], reduced_set[:,1],  c=colours,alpha=1,s=80)
#Plot.show()





#estrellas=np.array(estrellas)
#with open("/home/joaquintz/Desktop/facultad/datos/tsne_10k_random.json", 'w') as f:
#    json.dump(reduced_set,f,cls=NumpyEncoder)

#with open("/home/joaquintz/Desktop/facultad/datos/tsne_10k_estrellas.json", 'w') as f:
#    json.dump(estrellas,f,cls=NumpyEncoder)
