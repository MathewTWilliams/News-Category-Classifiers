#Author: Matt Williams
#Version: 12/02/2021

from sklearn.metrics import silhouette_score, rand_score, homogeneity_completeness_v_measure, \
normalized_mutual_info_score, adjusted_mutual_info_score

from classifier_metrics import calculate_classifier_metrics

import os
from constants import get_result_path, make_result_path
from numpy import NINF, PINF
from save_load_json import load_json, save_json



def calculate_cluster_metrics(true_labels, predictions, model_details, features):
    '''Given the true_labels, the predictions of the clustering algorithm, the clustering
    algorithm details, and the data used for the clustering, calculate the primary metrics 
    for clustering and save them to a json file along with the model details.'''
    h, c , v = homogeneity_completeness_v_measure(true_labels, predictions)
    model_details['Silhouette_Score'] = silhouette_score(features, predictions)
    model_details['Rand_Score'] = rand_score(true_labels, predictions)
    model_details['NMI'] = normalized_mutual_info_score(true_labels, predictions)
    model_details['AMI'] = adjusted_mutual_info_score(true_labels, predictions)
    model_details['Homogeneity'] = h
    model_details['Completeness'] = c
    model_details['V_Measure'] = v

    model_details['Predictions'] = predictions.tolist()
    file_path = make_result_path(model_details['Model'], model_details['Vector_Model'])
    save_json(model_details, file_path)

    


def find_best_result(clustering_name, vec_model_name, metric, large = True): 
    '''Given the clustering algorithm name, the vector model used, and the name of the clustering metric, 
    return the file name that contains the best metric value for the given parameters and that metric value.'''
    best_score = NINF if large else PINF
    best_file_name = ""
    for file in os.listdir(get_result_path(clustering_name,"")):
        if file.startswith(vec_model_name):
            results = load_json(get_result_path(clustering_name, file))
            if metric not in results.keys(): 
                return ""
            if large and results[metric] > best_score:
                best_file_name = file
                best_score = results[metric] 
            elif not large and results[metric] < best_score: 
                best_file_name = file
                best_score = results[metric]
    
    return best_file_name, best_score