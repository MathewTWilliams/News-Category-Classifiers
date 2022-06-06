#Author: Matt Williams
#Version: 6/04/2022

from sklearn.neighbors import NearestCentroid
from constants import ClassificationModels, WordVectorModels
from run_classification import run_classifier

near_cent_param_grid = {
    "metric" : ["cosine", "euclidean", "manhattan"], 
    "shrink_threshold" : list(range(0.1, 0.6, 0.1))
}


def run_near_centroid(vec_model_name, metric = "euclidean", shrink_threshold = None): 
    
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