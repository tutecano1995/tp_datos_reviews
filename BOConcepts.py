from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import csv
import random
import numpy as np

model = Word2Vec.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary = True)
words = set()
reviews = {}
with open('train.csv','r') as train_file:
	train_csv = csv.reader(train_file)
	next(train_csv)
	for row in train_csv:
			reviewid = row[0]
			predict = row[6]
			text = row[9]
			reviews[reviewid] = {}
			reviews[reviewid]['text'] = text
			reviews[reviewid]['pred'] = predict
			text_words = text.split(" ")
			if not(21<len(text_words)<223): continue
			for word in text_words:
					words.add(word)

selected_vectors = np.empty((0,300))	#La longitud de los vectores en el modelo es de 300
word_index = []
i = 0
for word in words:
	try:
		word_vector = model[word]
		selected_vectors = np.vstack((selected_vectors,word_vector))
		word_index.append(word)
	except:
		i+=1
		if not i%10000:	#Imprimo la cantidad de palabras "ausentes" cada mil
			print i


#NO SOMOS COMO SERVETTO, EL 99% DEL CODIGO ES DE ACA:
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

num_clusters = 100
kmeans_clustering = KMeans (n_clusters = num_clusters)
idx = kmeans_clustering.fit_predict(selected_vectors)
word_centroid_map = dict(zip(word_index,idx))

#Imprimo algunos clusters para ver que onda
for cluster in xrange(0,10):
	print "\nCluster %d" % cluster
	cluster_words = []
	for i in xrange(0,len(word_centroid_map.values())):
			if(word_centroid_map.values()[i] == cluster):
				cluster_words.append(word_centroid_map.keys()[i])
	print cluster_words


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


for reviewid, review in reviews.iteritems():
	review['vec'] = create_bag_of_centroids(review['text'].split(" "),word_centroid_map)

with open("BOC_review_"+str(num_clusters)+".json", 'w') as f:
    json.dump(review,f)

