from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from featurehash_y_tsne import *
import matplotlib as mpl
mpl.use('TkAgg')
import pylab as Plot
import csv
import random
import numpy as np
import json
import base64
import xgboost as xgb


def create_bag_of_centroids( wordlist, word_centroid_map ):
	#
	# The number of clusters is equal to the highest cluster index
	# in the word / centroid map
	num_centroids = max( word_centroid_map.values() ) + 1
	#
	# Pre-allocate the bag of centroids vector (for speed)
	bag_of_centroids = np.zeros( num_centroids, dtype="float32" )
	#
	# Loop over the words in the review. If the word is in the vocabulary,
	# find which cluster it belongs to, and increment that cluster count 
	# by one
	for word in wordlist:
		if word in word_centroid_map:
			index = word_centroid_map[word]
			bag_of_centroids[index] += 1
	#
	# Return the "bag of centroids"
	return bag_of_centroids

#Funcion muy cabeza que sirve para seleccionar aproximadamente 1500 de cada valor y que sirve solo para este caso, y que aproxima a una seleccion homogenea
#Le otorgo probabilidad 1500/#reviews_con_esa_estrella dependiendo de que estrella toca
#Para este caso el total de reviews por puntuacion es: [ 37899,  21576,  30305,  56900, 260571]
def elegir_filas_estrellas_homogeneas_2(estrellas):
    contador_estrellas=np.array([0]*5)
    cant_reviews=len(estrellas)
    filas = []
    total = 0
    for i in xrange(len(estrellas)):
    	rand = random.uniform(0,1)
    	if int(estrellas[i]) == 1:
    		if rand<0.039: filas.append(i)
    	if int(estrellas[i]) == 2:
    		if rand<0.069: filas.append(i)
    	if int(estrellas[i]) == 3:
    		if rand<0.049: filas.append(i)
    	if int(estrellas[i]) == 4:
    		if rand<0.026: filas.append(i)
    	if int(estrellas[i]) == 5:
    		if rand<0.0057: filas.append(i)    		
    return filas

model = Word2Vec.load_word2vec_format("data/GoogleNews-vectors-negative300.bin",binary = True)
words = set()
reviews = {}
i = 0
with open('train_processed.csv','r') as train_file:
	train_csv = csv.reader(train_file)
	next(train_csv)
	for row in train_csv:
			reviewid = row[1]
			predict = row[7]
			text = row[10]
			text_words = text.split(" ")
			if not(21<len(text_words)<223): continue
			reviews[reviewid] = {}
			reviews[reviewid]['text'] = text
			reviews[reviewid]['pred'] = predict
			if not i%10000: print(i)
			i+=1
			for word in text_words:
					words.add(word)

print("train_processed.csv cargado!")

test_reviews = {}

with open('test_processed.csv','r') as test_file:
	test_csv = csv.reader(test_file)
	next(test_csv)
	for row in test_csv:
			reviewid = row[1]
			text = row[9]
			text_words = text.split(" ")
			if not(21<len(text_words)<223): continue
			test_reviews[reviewid] = {}
			test_reviews[reviewid]['text'] = text
			if not i%10000: print(i)

print("test_processed.csv cargado!")



#Word embeddings
selected_vectors = np.empty((0,300))	#La longitud de los vectores en el modelo es de 300
word_index = []
i = 0
print("Comenzando filtrado de palabras")
for word in words:
	try:
		word_vector = model[word]
		selected_vectors = np.vstack((selected_vectors,word_vector))
		word_index.append(word)
	except:
		i+=1
		if not i%10000:	#Imprimo la cantidad de palabras "ausentes" cada mil
			print(i)


#NO SOMOS COMO SERVETTO, EL 99% DEL CODIGO ES DE ACA:
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

print("Clustering - KMeans")
num_clusters = 50
kmeans_clustering = KMeans (n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(selected_vectors)
word_centroid_map = dict(zip(word_index,idx))

#Imprimo algunos clusters para ver que onda
print("Imprimimos algunos clusters")
for cluster in xrange(0,3):
	print ("\nCluster %d" % cluster)
	cluster_words = []
	for i in xrange(0,len(word_centroid_map.values())):
			if(word_centroid_map.values()[i] == cluster):
				cluster_words.append(word_centroid_map.keys()[i])
	print (cluster_words)

print("Hacemos BOCs de train")
i = 0
train_matriz = []
estrellas_bocs = []
for reviewid, review in reviews.iteritems():
	review['vec'] = list(create_bag_of_centroids(review['text'].split(" "),word_centroid_map))
	i+=1
	if not i%10000:	print (i) 
	train_matriz.append(review['vec'])
	estrellas_bocs.append(int(review['pred']))

i = 0
print("Hacemos BOCs de test")
for reviewid, review in test_reviews.iteritems():
	review['vec'] = list(create_bag_of_centroids(review['text'].split(" "),word_centroid_map))
	i+=1
	if not i%10000:	print (i) 

print("No hacemos TSNE")

#filas = elegir_filas_estrellas_homogeneas_2(estrellas_bocs)
#estrellas_seleccionadas = [int(estrellas_bocs[k]) for k in filas]
#matriz_red = matriz_bocs[filas]

#colourmap= {1:'#011f4b',2:'#03396c',3:'#005b96',4:'#6497b1',5:'#b3cde0'}
#colours = [colourmap[n] for n in estrellas_seleccionadas]

#print("Y ahora reducimos con tsne")
#reduced_set =tsne(matriz_red,no_dims=2,initial_dims=20,perplexity=20.0,metric="cosine")



#star_hist=np.array([0]*5)
#for j in estrellas_seleccionadas:
#    star_hist[j-1]+=1

#print(star_hist)

#Plot.scatter(reduced_set[:,0],reduced_set[:,1],c=colours,alpha=1,s=80,marker='.',lw=0)
#Plot.show()

#estrellas_seleccionadas_save=np.array(estrellas_seleccionadas)

#try:
#	with open("tsne_10k_homo_cos_300cl.json", 'w') as f:
#    	json.dump(reduced_set,f,cls=NumpyEncoder)

#	with open("tsne_10k_homo_estrellas_300cl.json", 'w') as f:
#    	json.dump(estrellas_seleccionadas_save,f,cls=NumpyEncoder)

#except:
#	print("No se pudieron guardar los modelos")

train_matriz = np.matrix(train_matriz)
dtrain = xgb.DMatrix(train_matriz)
param = {'max_depth':10,'eta':0.2,'silent':0,'objective':'reg:linear'}
dtrain = xgb.DMatrix(train_matriz,label = estrellas_bocs)
bst = xgb.train(param,dtrain,num_boost_round = 3000)

print("Guardamos los modelos")

#Convierto las componentes de los vectores a int para poder guardarlo en json

for reviewid,review in reviews.iteritems():
		for i in range(len(review['vec'])):
			review['vec'][i] = int(review['vec'][i])

for reviewid,review in test_reviews.iteritems():
		for i in range(len(review['vec'])):
			review['vec'][i] = int(review['vec'][i])


try: 
	with open(str(num_clusters)+"_reg_xgb.csv", 'w') as f:
		f_writer = csv.writer(f)
		f_writer.writerow(['Id','Prediction'])
		for reviewid,review in test_reviews.iteritems():
			f_writer.writerow([reviewid,bst2.predict(xgb.DMatrix(review['vec']))[0]])

	with open("boc_"+str(num_clusters)+"clu_clean_train.json",'w') as f:
		json.dump(reviews,f)

	with open("boc_"+str(num_clusters)+"clu_clean_test.json",'w') as f:
		json.dump(test_reviews,f)

except:
	print("No se pudieron guardar los modelos")


