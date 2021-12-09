#Author: Matt Williams
#Version: 12/08/2021
#Reference: https://datatofish.com/line-chart-python-matplotlib/


import matplotlib.pyplot as plt
from save_load_json import load_json
from constants import get_result_path
from get_vec_models import get_vec_model_names
import os

def show_line_graph(x_labels, y_values, x_title, y_title, graph_title): 
    '''Given the name of the x axis labels, the values in the y axis, the title of the 
    x and y axes, and the title of the graph: make a line graph and display it to the user'''

    plt.figure(figsize=(12,12))
    plt.plot(x_labels, y_values, color = 'blue', marker = 'o')
    plt.title(graph_title, fontsize = 14)
    plt.xlabel(x_title, fontsize = 14)
    for label in plt.gca().get_xticklabels(): 
        label.set_rotation(30)
        label.set_fontsize(9)
    plt.ylabel(y_title, fontsize = 14)
    plt.grid(True)
    plt.show()


def find_results_file(classifier, vec_model):
    '''Given a classifier, and the vector model name, find
    the JSON file that contains the results assciated with the classifier
    and vector model.'''
    for file in os.listdir(get_result_path(classifier,'')): 
         if file.startswith(vec_model): 
             return get_result_path(classifier,file)


if __name__ == "__main__":

    metrics = ['precision', 'recall', 'f1-score', 'accuracy']


    x_labels = ['Gaussian Naive Bayes', 'K-Nearest Neighbor', 'Logistic Regression', \
        'Multi-Layer Perceptron', 'Random Forest', 'Support Vector Machine']

    #For each vector model, make a line graph for each metric above.
    for vec_model_name in get_vec_model_names(): 
        y_values = {metric: [] for metric in metrics}

        for classifier in x_labels: 
            file_path = find_results_file(classifier, vec_model_name)
            results = load_json(file_path)
            
            for metric in metrics:
                if metric == 'accuracy': 
                    y_values[metric].append(results['Classification_Report']['accuracy'])
                else: 
                    y_values[metric].append(results['Classification_Report']['weighted avg'][metric])
        

        for metric in metrics: 
            show_line_graph(x_labels, y_values[metric], "Classifiers", metric +" values", \
                "Classifier " + metric.capitalize() + " values with " + vec_model_name.capitalize())
                
