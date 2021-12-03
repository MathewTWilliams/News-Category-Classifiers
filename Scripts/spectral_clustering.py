#Author: Matt Williams
#Version: 12/02/2021
from sklearn.cluster import SpectralClustering
from get_article_vectors import get_combined_train_test_info
from clustering_metrics import calculate_cluster_metrics
from constants import CATEGORIES, RAND_STATE, convert_categories_to_numbers
import pandas as pd
from visualize_article_vecs import visualize_article_vecs
from classifier_metrics import find_best_result




def run_spectral_clustering(vec_model_name, affinity = "rbf"): 
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the spectral clustering algorithm and save the results to a json file.'''

    data, labels = get_combined_train_test_info(vec_model_name)

    sc = SpectralClustering(affinity=affinity, n_clusters=len(CATEGORIES), random_state=RAND_STATE)


    predictions = sc.fit_predict(data)

    labels = convert_categories_to_numbers(labels)
    


    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : "Spectral_Clustering",
        'Affinity' : affinity, 
    }

    calculate_cluster_metrics(labels, predictions, model_details, data)
    return pd.Series(predictions)


if __name__ == "__main__": 

    #run_spectral_clustering("word2vec")
    #run_spectral_clustering("fasttext")
    #run_spectral_clustering("glove")

    #run_spectral_clustering("word2vec", affinity='nearest_neighbors')
    #run_spectral_clustering("fasttext", affinity='nearest_neighbors')
    #run_spectral_clustering("glove", affinity='nearest_neighbors')

    #run_spectral_clustering("word2vec", affinity='linear')
    ft_preds = run_spectral_clustering("fasttext", affinity='linear')
    visualize_article_vecs('fasttext', 2, ft_preds)
    #run_spectral_clustering("glove", affinity='linear')

    w2v_preds = run_spectral_clustering("word2vec", affinity='poly')
    visualize_article_vecs("word2vec", 2, w2v_preds)
    #run_spectral_clustering("fasttext", affinity='poly')
    #run_spectral_clustering("glove", affinity='poly')

    #run_spectral_clustering("word2vec", affinity='polynomial')
    #run_spectral_clustering("fasttext", affinity='polynomial')
    #run_spectral_clustering("glove", affinity='polynomial')

    #run_spectral_clustering("word2vec", affinity='laplacian')
    #run_spectral_clustering("fasttext", affinity='laplacian')
    #run_spectral_clustering("glove", affinity='laplacian')

    #run_spectral_clustering("word2vec", affinity='sigmoid')
    #run_spectral_clustering("fasttext", affinity='sigmoid')
    #run_spectral_clustering("glove", affinity='sigmoid')

    #run_spectral_clustering("word2vec", affinity='cosine')
    #run_spectral_clustering("fasttext", affinity='cosine')
    glv_preds = run_spectral_clustering("glove", affinity='cosine')
    visualize_article_vecs('glove', 2, glv_preds)

    #print("Word2Vec:",find_best_result("Spectral_Clustering", "word2vec", "accuracy"))
    #print("FastText:", find_best_result("Spectral_Clustering", "fasttext", "accuracy"))
    #print("Glove:", find_best_result("Spectral_Clustering", "glove", "accuracy"))