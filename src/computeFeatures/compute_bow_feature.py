# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import cluster_util
import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import app
import utils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix


##########################################################################################
# compute_local_segment_extraction
##########################################################################################
def apply_compute_local_segment_extraction(cluster, X_test=False):
    if X_test:
        cluster_util.apply_cores_function(app.script_bow_local_segment_extraction, app.X_test_path_ext_tmp, cluster, app.bow_X_test_segments)    
    else:
        cluster_util.apply_cores_function(app.script_bow_local_segment_extraction, app.X_train_path_ext_tmp, cluster, app.bow_X_train_segments)    


##########################################################################################
# KMEANS 1
##########################################################################################
def apply_kmeans_codewords():
    startTime = time.time()
    X=[]
      
    for sample in utils.array_segments_to_matrix(np.array(np.loadtxt(app.bow_X_train_segments, delimiter=";")), app.wave_level):
        for segment in sample:
            X.append(segment)
    X = np.array(X)
    
    print 'points : ', X.shape
    
    kmeans = MiniBatchKMeans(init='k-means++', n_clusters=app.codebook_size, batch_size=100, init_size=3*app.codebook_size, n_init=20)
    kmeans.fit(X)
    k_means_labels = kmeans.labels_
    k_means_cluster_centers = kmeans.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)
    
    np.savetxt(app.bow_cdw_file, k_means_cluster_centers, fmt='%.8e', delimiter=";")
    print 'number of cdw : ', len(k_means_labels_unique)
    print 'finish in ', str(time.time() - startTime), ' seconds'
    
##########################################################################################
# Assigne codewords and compute histogram
##########################################################################################
def apply_assign_codewords(cluster, X_test=False):
    if X_test:
        cluster_util.apply_cores_function(app.script_bow_histogram, app.bow_X_test_segments, cluster, app.bow_X_test_histo)    
    else:
        cluster_util.apply_cores_function(app.script_bow_histogram, app.bow_X_train_segments, cluster, app.bow_X_train_histo)    
    
##########################################################################################
# Classification
##########################################################################################

def apply_classification(CV=False):
    y_train = np.array(pd.read_csv(app.y_train_path_ext_tmp, sep=";", header=None)[0])
    # features
    X_train_histogram_txt = np.array(np.loadtxt(app.bow_X_train_histo, delimiter=";"))
    print X_train_histogram_txt.shape
    XX_train = np.c_[X_train_histogram_txt]
    
    if CV:
        XX_train_used_min, XX_test_used_min, y_train_used_min, y_test_used_min = train_test_split(XX_train, y_train, random_state=42, test_size=0.2)
        print XX_train_used_min.shape, XX_test_used_min.shape, y_train_used_min.shape, y_test_used_min.shape
        
        # KNeighborsClassifier
        grid = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors':[1, 10, 20, 30]}, cv=3)
        grid.fit(XX_train_used_min, y_train_used_min)
        print 'KNeighborsClassifier : ', grid.grid_scores_
        scores = [x[1] for x in grid.grid_scores_]
        print scores
        
        grid = GridSearchCV(linear_model.LogisticRegression(), param_grid={'penalty':['l2'], 'C':[10,30]}, cv=3)
        grid.fit(XX_train_used_min, y_train_used_min)
        print 'LogisticRegression : ', grid.grid_scores_
        scores = [x[1] for x in grid.grid_scores_]
        print scores
    
        grid = GridSearchCV(RandomForestClassifier(), param_grid={}, cv=3)
        grid.fit(XX_train_used_min, y_train_used_min)
        print 'RandomForestClassifier : ', grid.grid_scores_
        scores = [x[1] for x in grid.grid_scores_]
        print scores
        
        grid = GridSearchCV(GaussianNB(), param_grid={}, cv=3)
        grid.fit(XX_train_used_min, y_train_used_min)
        print 'GaussianNB : ', grid.grid_scores_
        scores = [x[1] for x in grid.grid_scores_]
        print scores
        
        y_pred = grid.predict(XX_test_used_min)
        cm = confusion_matrix(y_test_used_min, y_pred)
        plt.matshow(cm)
        plt.show()
        print cm
        
    else:
        X_test_histogram_txt = np.array(np.loadtxt(app.bow_X_test_histo, delimiter=";"))
        print X_test_histogram_txt.shape
        XX_test = np.c_[X_test_histogram_txt]
        clf = KNeighborsClassifier(n_neighbors=20)
        clf.fit(XX_train, y_train)
        y_pred = clf.predict(XX_test)
        save_predict(y_pred)
        

def save_predict(y_pred):
	np.savetxt(app.y_test_path_ext_tmp, y_pred, fmt='%s')
	print y_pred[:100]  
 

if __name__ == '__main__':

    # get workers computer from all the local network
    cluster = cluster_util.get_cluster(app.regex_network, 2)
    print cluster
    
    utils.write_dataset_elements()
    
    apply_compute_local_segment_extraction(cluster)
    apply_kmeans_codewords()
    apply_assign_codewords(cluster)
    apply_classification(CV=True)
    
    apply_compute_local_segment_extraction(cluster, X_test=True)
    apply_assign_codewords(cluster, X_test=True)
    apply_classification()