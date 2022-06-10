#Author: Matt Williams
#Version: 6/04/2022

from sklearn.neighbors import RadiusNeighborsClassifier
from constants import ClassificationModels, WordVectorModels
from run_classification import run_classifier
import numpy as np

# Parameter grid for cross validation
near_rad_param_grid = {
    "radius" : np.arange(0.5, 3.0, 0.5).tolist(), #here
    "algorithm" : ["ball_tree, kd_tree"], 
    "leaf_size" : list(range(10, 60, 10)), 
    "p" : list(range(1,5)), 
    "metric" : ["minkowski"], 
    "weights" : ["distance"]
}


def run_near_radius(vec_model_name, radius = 1.0, algorithm = "auto", leaf_size = 30, \
                p = 2, metric = "minkowski"): 
    
    near_radius = RadiusNeighborsClassifier(radius=radius, algorithm=algorithm, leaf_size=leaf_size, \
                                        p=p, metric=metric, weights="distance")

    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.RAD.value, 
        "radius" : radius, 
        "algorithm" : algorithm, 
        "leaf_size" : leaf_size, 
        "p" : p, 
        "metric" : metric, 
        "weights" : "distance"
    }

    run_classifier(vec_model_name, near_radius, model_details)

if __name__ == "__main__": 
    run_near_radius(WordVectorModels.WORD2VEC.value)
    run_near_radius(WordVectorModels.FASTTEXT.value)
    run_near_radius(WordVectorModels.GLOVE.value)