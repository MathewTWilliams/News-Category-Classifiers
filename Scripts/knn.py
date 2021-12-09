#Author: Matt Williams
#Version: 12/08/2021
from sklearn.neighbors import KNeighborsClassifier
from get_article_vectors import get_test_info, get_training_info
from make_confusion_matrix import show_confusion_matrix
from classifier_metrics import calculate_classifier_metrics


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
    show_confusion_matrix(test_labels, predictions, "KNN w/" + vec_model_name + " Confusion Matrix")
        
        

if __name__ == "__main__": 
    
    run_knn("word2vec", n_neighbors=10, weights='distance', algorithm='ball_tree', p=1)
    run_knn("fasttext", n_neighbors=8, weights='distance', algorithm='ball_tree', p=1)
    run_knn("glove", n_neighbors=10, weights='distance', algorithm='ball_tree', p = 3)
