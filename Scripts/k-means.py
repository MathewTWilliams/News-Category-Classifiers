#Author: Matt Williams
#Version: 12/02/2021
from sklearn.cluster import KMeans
from get_article_vectors import get_combined_train_test_info
from constants import CATEGORIES, RAND_STATE, convert_categories_to_numbers
from clustering_metrics import calculate_cluster_metrics
from classifier_metrics import find_best_result
from visualize_article_vecs import visualize_article_vecs
import pandas as pd



def run_k_means(vec_model_name, n_init = 10, tol = 1e-4):
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the K-means clustering algorithm and save the results to a json file.'''
    data, labels = get_combined_train_test_info(vec_model_name)


    labels = convert_categories_to_numbers(labels)
    



    k_means = KMeans(n_clusters=len(CATEGORIES), algorithm="full", 
                    n_init= n_init, tol = tol, random_state=RAND_STATE)

    predictions = k_means.fit_predict(data)


    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : "K_Means",
        'Number runs with different initial Centroid Seeds' : n_init, 
        'Tolerance to Froenius Norm' : tol

    }
    calculate_cluster_metrics(labels, predictions, model_details, data)
    return pd.Series(predictions)


if __name__ == "__main__": 
    #uncommented lines had best accuracy
    
    #default settings
    w2v_preds = run_k_means("word2vec") 
    visualize_article_vecs("word2vec", 2, w2v_preds)
    #run_k_means("fasttext")
    #run_k_means("glove")

    #run_k_means("word2vec", tol = 1e-3)
    #run_k_means("fasttext", tol = 1e-3)
    #run_k_means("glove", tol = 1e-3) 

    #run_k_means("word2vec", tol = 1e-5)
    #run_k_means("fasttext", tol = 1e-5) 
    #run_k_means("glove", tol = 1e-5)

    #run_k_means("word2vec", n_init = 5)
    ft_preds = run_k_means("fasttext", n_init = 5)
    visualize_article_vecs("fasttext", 2, ft_preds)
    #run_k_means("glove", n_init = 5)

    #run_k_means("word2vec", n_init = 15)
    #run_k_means("fasttext", n_init = 15)
    glv_preds = run_k_means("glove", n_init = 15)
    visualize_article_vecs("glove", 2, glv_preds)


    #run_k_means("word2vec", tol = 1e-3, n_init = 5)
    #run_k_means("fasttext", tol = 1e-3, n_init = 5)
    #run_k_means("glove", tol = 1e-3, n_init = 5)

    #run_k_means("word2vec", tol = 1e-5, n_init = 5)
    #run_k_means("fasttext", tol = 1e-5, n_init = 5)
    #run_k_means("glove", tol = 1e-5, n_init = 5)

    #run_k_means("word2vec", tol = 1e-3, n_init = 15)
    #run_k_means("fasttext", tol = 1e-3, n_init = 15)
    #run_k_means("glove", tol = 1e-3, n_init = 15)

    #run_k_means("word2vec", tol = 1e-5, n_init = 15)
    #run_k_means("fasttext", tol = 1e-5, n_init = 15)
    #run_k_means("glove", tol = 1e-5, n_init = 15)

    

    print("Word2Vec:",find_best_result("K_Means", "word2vec", "accuracy"))
    print("FastText:", find_best_result("K_Means", "fasttext", "accuracy"))
    print("Glove:", find_best_result("K_Means", "glove", "accuracy"))
