#Author: Matt Williams
#Version: 6/04/2022

from sklearn.neighbors import NearestCentroid
from utils import ClassificationModels, WordVectorModels
from run_classification import run_classifier
import numpy as np

# Parameter grid for cross validation
near_cent_param_grid = {
    "metric" : ["cosine", "euclidean", "manhattan"], 
    "shrink_threshold" : np.arange(0.1, 0.6, 0.1).tolist()
}


def run_near_centroid(vec_model_name, metric = "euclidean", shrink_threshold = None): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Nearest Centroid Classification algorithm and save the results to a json file.'''
    near_cent = NearestCentroid(metric=metric, shrink_threshold=shrink_threshold)

    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.CENT.value, 
        "metric" : metric, 
        "shrink_threshold" : shrink_threshold,
    }

    run_classifier(vec_model_name, near_cent, model_details)


if __name__ == "__main__": 
    run_near_centroid(WordVectorModels.WORD2VEC.value)
    run_near_centroid(WordVectorModels.FASTTEXT.value)
    run_near_centroid(WordVectorModels.GLOVE.value)