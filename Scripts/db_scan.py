#Author: Matt Williams
#Version: 12/02/2021
#Couldn't get DBSCAN to work correctly with the Word2Vec vectors as input.


from sklearn.cluster import DBSCAN
from get_article_vectors import get_training_info, get_test_info
from get_vec_models import get_vec_model_names
from clustering_metrics import calculate_cluster_metrics
from constants import convert_categories_to_numbers


def run_db_scan(vec_model_name, eps = 0.5, min_samples = 5, metric = 'cosine'): 

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the db scan clustering algorithm and save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    dbs = DBSCAN(eps=eps, min_samples=min_samples, metric= metric)
    
    predictions = dbs.fit_predict(test_data)

    test_labels = convert_categories_to_numbers(test_labels)

    model_details = {
        'Vector_Model': vec_model_name,
        'Model' : 'DB_SCAN', 
        'eps' :  eps, 
        'min_samples': min_samples, 
        'metric' : metric, 
    }

    calculate_cluster_metrics(test_labels, predictions , model_details, test_data)




if __name__ == "__main__":
    
    
    run_db_scan("word2vec", eps = 0.1, min_samples=30)
    run_db_scan("fasttext", eps = 0.1, min_samples=30)
    run_db_scan("glove", eps = 0.1, min_samples=30)

