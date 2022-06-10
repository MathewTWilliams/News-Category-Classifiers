#Author: Matt Williams
#Version: 12/08/2021
from sklearn.neighbors import KNeighborsClassifier
from constants import WordVectorModels, ClassificationModels
from run_classification import run_classifier

#Parameter grid for Cross Validation
knn_param_grid = {
    "n_neighbors" : list(range(5, 30, 5)),
    "weights" : ['uniform', 'distance'], 
    "algorithm" : ['ball_tree', 'kd_tree'],
    "p" : list(range(1,5)),
    "leaf_size" : list(range(10, 60, 10))

}


def run_knn(vec_model_name, n_neighbors = 5, weights = 'uniform', 
            algorithm = 'auto', p = 2, leaf_size = 30):
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the K-Nearest Neighbors Classification algorithm and save the results to a json file.'''
   
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, 
                                p=p, leaf_size=leaf_size)
                                
    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.KNN.value,
        'N_Neighbors': n_neighbors, 
        "Weights" : weights,
        "algorithm" : algorithm, 
        "p" : p, 
        "leaf_size" : leaf_size
    }

    run_classifier(vec_model_name, knn, model_details)
        
        

if __name__ == "__main__": 
    
    run_knn(WordVectorModels.WORD2VEC.value, n_neighbors=10, weights='distance', algorithm='ball_tree', p=1)
    run_knn(WordVectorModels.FASTTEXT.value, n_neighbors=8, weights='distance', algorithm='ball_tree', p=1)
    run_knn(WordVectorModels.GLOVE.value, n_neighbors=10, weights='distance', algorithm='ball_tree', p = 3)
