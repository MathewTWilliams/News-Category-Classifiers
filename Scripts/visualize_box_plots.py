#Author: Matt Williams
#Version: 12/08/2021


import matplotlib.pyplot as plt 
from constants import get_cv_result_path, K_FOLDS
from save_load_json import load_json
import os
from get_vec_models import get_vec_model_names


accuracy_key_suffix = "_test_accuracy"
precision_key_suffix = "_test_precision_weighted"
recall_key_suffix = "_test_recall_weighted"
f1_key_suffix = "_test_f1_weighted"

params_key = "params"



def make_box_plot(title, x_values): 
    '''Given the title of the box plot, the x-axis and y-axis names, and the x_values
    show a box plot of the given information. '''
    
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.boxplot(x_values)
    plt.show()




def show_classifier_box_plots(vec_model_name, classifier_name): 
    '''Given a vector model name and the name of the classifier, 
    show a box plot visual of the accuracy, precision, recall, and f1 score for that
    vector model and classifier from the grid search cross validation. 
    Splits within the top 10 accuracies will only be shown in the box plot. '''
    results_dir = get_cv_result_path(classifier_name,"")
    num_splits = 10

    if not os.path.exists(results_dir): 
        return 

    file_name = ""
    for file in os.listdir(results_dir): 
        if file.startswith(vec_model_name):
            file_name = file
            break

    results = load_json(get_cv_result_path(classifier_name, file_name))

    split_indicies = [] 
    #Grab the indices of the top 10 best splits
    for index, value in enumerate(results['rank' + accuracy_key_suffix]):
        if value in range(1, num_splits+1): 
            split_indicies.append(index)
            if len(split_indicies) == num_splits:
                break



    #value is a 2d list where each row represents a param split and 
    #each column in row is a value for specified key
    #Example: results_subset['accuracy'][i][j]
    # represents the jth accuracy for the ith param split
    results_subset = {
        'accuracy': [[] for _ in range(num_splits)],
        'precision': [[] for _ in range(num_splits)],
        'recall': [[] for _ in range(num_splits)],
        'f1 score': [[] for _ in range(num_splits)]
    }

    #grid validation results stores results by Kth split of cross validation, then by param split
    #Example: results[split0_test_accuracy] is a list of accuracies for the 1st fold.
    #Example: results[split0_test_accuracy][0] is the accuracy of the 1st param split in the 1st fold. 
    for i in range(K_FOLDS):
        key_prefix = "split" + str(i)
        for j, value in enumerate(split_indicies):
            results_subset["accuracy"][j].append(results[key_prefix + accuracy_key_suffix][value])
            results_subset['precision'][j].append(results[key_prefix + precision_key_suffix][value])
            results_subset['recall'][j].append(results[key_prefix + recall_key_suffix][value])
            results_subset["f1 score"][j].append(results[key_prefix + f1_key_suffix][value])


            
 
    for key in results_subset.keys(): 
        title = classifier_name + " "+ vec_model_name + " " + key +" box plot"
        make_box_plot(title, results_subset[key])

def show_naive_bayes_box_plots(vec_model_name):
    '''Show Box Plot visuals for Naive Bayes classifier'''
    show_classifier_box_plots(vec_model_name, "Gaussian Naive Bayes")

def show_svm_box_plots(vec_mode_name): 
    '''Show Box Plot visuals for Support Vector Machine Classifier'''
    show_classifier_box_plots(vec_mode_name, "Support Vector Machine")

def show_rf_box_plots(vec_mode_name): 
    '''Show Box Plot visuals for Support Vector Machine Classifier'''
    show_classifier_box_plots(vec_mode_name, "Random Forest")

def show_mlp_box_plots(vec_mode_name): 
    '''Show Box Plot visuals for Support Vector Machine Classifier'''
    show_classifier_box_plots(vec_mode_name, "Multi-Layer Perceptron")

def show_log_regr_box_plots(vec_mode_name): 
    '''Show Box Plot visuals for Support Vector Machine Classifier'''
    show_classifier_box_plots(vec_mode_name, "Logistic Regression")

def show_knn_box_plots(vec_mode_name): 
    '''Show Box Plot visuals for Support Vector Machine Classifier'''
    show_classifier_box_plots(vec_mode_name, "K-Nearest Neighbor")

    

    
    

if __name__ == "__main__":

    for vec_model_name in get_vec_model_names(): 
        show_naive_bayes_box_plots(vec_model_name)
        show_knn_box_plots(vec_model_name)
        show_log_regr_box_plots(vec_model_name)
        show_mlp_box_plots(vec_model_name)
        show_rf_box_plots(vec_model_name)
        show_svm_box_plots(vec_model_name)

