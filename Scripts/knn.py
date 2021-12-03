#Author: Matt Williams
#Version: 12/02/2021
from sklearn.neighbors import KNeighborsClassifier
from get_article_vectors import get_test_info, get_training_info
from get_vec_models import get_vec_model_names
from classifier_metrics import calculate_classifier_metrics
from cross_validation import run_cross_validation

#Param grid for Grid Search Cross Validation
knn_param_grid = {
    "n_neighbors" : [5, 8, 10],
    "weights" : ['uniform', 'distance'], 
    "algorithm" : ['ball_tree', 'kd_tree'],
    "p" : [1,2,3],

}


def run_knn(vec_model_name, n_neighbors = 5, weights = 'uniform', 
            algorithm = 'auto', p = 2):
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the K-Nearest Neighbors Classification algorithm and save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, 
                                p=p)

    #run_cross_validation(knn, "K-Nearest Neighbor", vec_model_name)
    knn.fit(training_data, training_labels)
    predictions = knn.predict(test_data)

    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : "K-Nearest Neighbor",
        'N_Neighbors': n_neighbors, 
        "Weights" : weights,
        "algorithm" : algorithm, 
        "p" : p, 
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)

        
        

if __name__ == "__main__": 
    for vec_model_name in get_vec_model_names(): 
        run_knn(vec_model_name)
