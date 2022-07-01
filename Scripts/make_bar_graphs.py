#Author: Matt Williams
#Version: 06/29/2022

import matplotlib.pyplot as plt
from save_load_json import load_test_result
from utils import get_result_path, ClassificationModels, WordVectorModels, get_result_bar_graph_path
import os
import numpy as np

def show_bar_graph(y_labels, x_values, y_title, x_title, graph_title):

    min = x_values[0]
    max = x_values[0]

    for i in range(1, len(x_values)): 
        if x_values[i] < min: 
            min = x_values[i]
        elif x_values[i] > max: 
            max = x_values[i]

    y_pos = np.arange(len(y_labels)) 

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(20)
    ax.barh(y_pos, x_values, align = "center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(graph_title)
    plt.xlim([min - 0.1, max])
    plt.savefig(get_result_bar_graph_path(graph_title + ".png"))


if __name__ == "__main__": 

    classifiers = ClassificationModels.get_values_as_list()
    wv_models = WordVectorModels.get_values_as_list()

    classifier_acc_dict = {classifier:[] for classifier in classifiers}
    wv_model_acc_dict = {wv_model:[] for wv_model in wv_models}

    for wv_model in wv_models: 
        for classifier in classifiers:
            accuracy = load_test_result(classifier, wv_model)["Classification_Report"]["accuracy"]
            classifier_acc_dict[classifier].append(accuracy)
            wv_model_acc_dict[wv_model].append(accuracy)


    for classifier in classifiers:
        x_values = classifier_acc_dict[classifier]
        y_labels = wv_models
        show_bar_graph(y_labels, x_values, "Word Vector Models", "Accuracy", "{} accuracies".format(classifier))

    for wv_model in wv_models: 
        x_values = wv_model_acc_dict[wv_model]
        y_labels = classifiers
        show_bar_graph(y_labels, x_values, "Classifiers", "Accuracy", "{} accuracies".format(wv_model))

    
    


