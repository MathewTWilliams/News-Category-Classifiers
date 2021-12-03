#Author: Matt Williams
#Version: 12/02/2021
from sklearn.cluster import AgglomerativeClustering
from get_article_vectors import get_combined_train_test_info
from clustering_metrics import calculate_cluster_metrics, find_best_result
from constants import CATEGORIES, convert_categories_to_numbers
from visualize_article_vecs import visualize_article_vecs
import pandas as pd

def run_hierarchical_clustering(vec_model_name, linkage = "ward", affinity = 'euclidean'):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the hierarichal clustering algorithm and save the results to a json file.'''
    data, labels = get_combined_train_test_info(vec_model_name)
    ac = AgglomerativeClustering(n_clusters=len(CATEGORIES), linkage=linkage, affinity=affinity)

    predictions = ac.fit_predict(data)

    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : "Hierarchical_Clustering",
        'Linkage': linkage, 
        'Affinity' : affinity

    }


    labels = convert_categories_to_numbers(labels)

    calculate_cluster_metrics(labels, predictions, model_details, data)
    return pd.Series(predictions)
    


if __name__ == "__main__": 

    #uncommented ones have best silhouette score

    #default linkage = ward, affinity = euclidean
    #run_hierarchical_clustering("word2vec")
    #run_hierarchical_clustering("fasttext")
    #run_hierarchical_clustering("glove")

    #manhattan distance
    #run_hierarchical_clustering("word2vec", linkage = "single", affinity="manhattan")
    ft_preds = run_hierarchical_clustering("fasttext", linkage="single", affinity="manhattan")
    visualize_article_vecs("fasttext", 2, ft_preds)
    #run_hierarchical_clustering("glove", linkage="single", affinity="manhattan")

    #run_hierarchical_clustering("word2vec", linkage = "average", affinity="manhattan")
    #run_hierarchical_clustering("fasttext", linkage="average", affinity="manhattan")
    #run_hierarchical_clustering("glove", linkage="average", affinity="manhattan")

    #run_hierarchical_clustering("word2vec", linkage = "complete", affinity="manhattan")
    #run_hierarchical_clustering("fasttext", linkage="complete", affinity="manhattan")
    #run_hierarchical_clustering("glove", linkage="complete", affinity="manhattan")

    #euclidean distance
    #run_hierarchical_clustering("word2vec", linkage = "single", affinity="euclidean")
    #run_hierarchical_clustering("fasttext", linkage="single", affinity="euclidean")
    #run_hierarchical_clustering("glove", linkage="single", affinity="euclidean")

    w2v_preds = run_hierarchical_clustering("word2vec", linkage = "average", affinity="euclidean")
    visualize_article_vecs("word2vec", 2, w2v_preds)
    #run_hierarchical_clustering("fasttext", linkage="average", affinity="euclidean")
    glv_preds = run_hierarchical_clustering("glove", linkage="average", affinity="euclidean")
    visualize_article_vecs("glove", 2, glv_preds)

    #run_hierarchical_clustering("word2vec", linkage = "complete", affinity="euclidean")
    #run_hierarchical_clustering("fasttext", linkage="complete", affinity="euclidean")
    #run_hierarchical_clustering("glove", linkage="complete", affinity="euclidean")

    #cosine distance
    #run_hierarchical_clustering("word2vec", linkage = "single", affinity="cosine")
    #run_hierarchical_clustering("fasttext", linkage="single", affinity="cosine")
    #run_hierarchical_clustering("glove", linkage="single", affinity="cosine")

    #run_hierarchical_clustering("word2vec", linkage = "average", affinity="cosine")
    #run_hierarchical_clustering("fasttext", linkage="average", affinity="cosine")
    #run_hierarchical_clustering("glove", linkage="average", affinity="cosine")

    #run_hierarchical_clustering("word2vec", linkage = "complete", affinity="cosine")
    #run_hierarchical_clustering("fasttext", linkage="complete", affinity="cosine")
    #run_hierarchical_clustering("glove", linkage="complete", affinity="cosine")

    #file, result = find_best_result("Hierarchical_Clustering", "word2vec", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)
    #file, result = find_best_result("Hierarchical_Clustering", "fasttext", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)
    #file, result = find_best_result("Hierarchical_Clustering", "glove", "Silhouette_Score")
    #print(file, "Silhouette_Score:", result)

    #file, result = find_best_result("Hierarchical_Clustering", "word2vec", "NMI")
    #print(file, "NMI:", result)
    #file, result = find_best_result("Hierarchical_Clustering", "fasttext", "NMI")
    #print(file, "NMI:", result)
    #file, result = find_best_result("Hierarchical_Clustering", "glove", "NMI")
    #print(file, "NMI:", result)



