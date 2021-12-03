#Author: Matt Williams
#Version: 12/02/2021
from sklearn.cluster import SpectralClustering
from get_article_vectors import get_combined_train_test_info
from clustering_metrics import calculate_cluster_metrics, find_best_result
from constants import CATEGORIES, RAND_STATE, convert_categories_to_numbers
import pandas as pd
from visualize_article_vecs import visualize_article_vecs




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

    #uncommented ones have best silhouette score
    #run_spectral_clustering("word2vec")
    ft_preds = run_spectral_clustering("fasttext")
    visualize_article_vecs("fasttext",2,ft_preds)
    #run_spectral_clustering("glove")

    #run_spectral_clustering("word2vec", affinity='nearest_neighbors')
    #run_spectral_clustering("fasttext", affinity='nearest_neighbors')
    #run_spectral_clustering("glove", affinity='nearest_neighbors')

    #run_spectral_clustering("word2vec", affinity='linear')
    #run_spectral_clustering("fasttext", affinity='linear')
    #run_spectral_clustering("glove", affinity='linear')

    #run_spectral_clustering("word2vec", affinity='poly')
    #run_spectral_clustering("fasttext", affinity='poly')
    glv_preds = run_spectral_clustering("glove", affinity='poly')
    visualize_article_vecs("glove",2,glv_preds)

    #run_spectral_clustering("word2vec", affinity='polynomial')
    #run_spectral_clustering("fasttext", affinity='polynomial')
    #run_spectral_clustering("glove", affinity='polynomial')

    w2v_preds = run_spectral_clustering("word2vec", affinity='laplacian')
    visualize_article_vecs("word2vec",2,w2v_preds)
    #run_spectral_clustering("fasttext", affinity='laplacian')
    #run_spectral_clustering("glove", affinity='laplacian')

    #run_spectral_clustering("word2vec", affinity='sigmoid')
    #run_spectral_clustering("fasttext", affinity='sigmoid')
    #run_spectral_clustering("glove", affinity='sigmoid')

    #run_spectral_clustering("word2vec", affinity='cosine')
    #run_spectral_clustering("fasttext", affinity='cosine')
    #run_spectral_clustering("glove", affinity='cosine')


    #file, result = find_best_result("Spectral_Clustering", "word2vec", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)
    #file, result = find_best_result("Spectral_Clustering", "fasttext", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)
    #file, result = find_best_result("Spectral_Clustering", "glove", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)

    #file, result = find_best_result("Spectral_Clustering", "word2vec", "NMI")
    #print(file, "NMI:", result)
    #file, result = find_best_result("Spectral_Clustering", "fasttext", "NMI")
    #print(file, "NMI:", result)
    #file, result = find_best_result("Spectral_Clustering", "glove", "NMI")
    #print(file, "NMI:", result)

