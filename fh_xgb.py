from sklearn.feature_extraction import FeatureHasher
import csv
import numpy as np
import base64
import xgboost as xgb

h = FeatureHasher(n_features=25, input_type='string')

estrellas = []
print "Creating feature hashes"
with open('train_processed.csv','r') as train_file:
	train_csv = csv.reader(train_file)
	next(train_csv)
	reviews_train = []
	for row in train_csv:
		estrellas.append(int(row[7]))
		reviews_train.append(row[10])
	matriz_reviews_train = h.transform(reviews_train).toarray()
print "Created feature hashes"

print "Creating xgboost train set"
train_matriz = np.matrix(matriz_reviews_train)
dtrain = xgb.DMatrix(train_matriz)
param = {'max_depth':10,'eta':0.1,'silent':0,'objective':'reg:linear'}
dtrain = xgb.DMatrix(train_matriz,label = estrellas)
bst = xgb.train(param,dtrain,num_boost_round = 1000)
print "Created xgboost train set"

print "Creating final file"
with open('test_processed.csv','r') as test_file:
	test_csv=csv.reader(test_file)
	with open('submission.csv','w') as submission_file:
		submission=csv.writer(submission_file)
		next(test_csv)
		submission.writerow(["Id","Prediction"])
        
		for row in test_csv:
			reviews_test = [row[9]]
			data = h.transform(reviews_test).toarray()
			dtest = xgb.DMatrix(data)
			ypred = bst.predict(dtest)
			submission.writerow([row[1], ypred])

print "Created final file"
