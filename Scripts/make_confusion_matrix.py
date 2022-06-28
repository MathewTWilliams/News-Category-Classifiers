#Author: Matt Williams
#Version: 12/06/2021
#Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
####

from msilib.schema import Class
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from save_load_json import load_test_result
from utils import ClassificationModels, WordVectorModels, get_result_visual_path
from get_article_vectors import get_test_info
from save_load_json import load_test_result


def show_confusion_matrix(true_labels, predictions, title):
    '''Given the true_labels, and the predictions, make and show a confusion matrix'''
    _, ax = plt.subplots(figsize=(12,12)) 

    disp = ConfusionMatrixDisplay.from_predictions(true_labels, predictions, normalize='true', 
                                                    cmap=plt.cm.Blues, ax=ax)


    x_ticklabels = ax.get_xticklabels()
    for label in x_ticklabels: 
        label.set_rotation(45)
    disp.ax_.set_title(title)
    np.set_printoptions(precision = 2)
    plt.savefig(get_result_visual_path(title + ".png"))


if __name__ == "__main__":


    for wv_model in WordVectorModels.get_values_as_list(): 
        _ , test_labels = get_test_info(wv_model)
        for classifier in ClassificationModels.get_values_as_list():

            #these classifiers were scrapped
            if classifier == ClassificationModels.RAD.value or classifier == ClassificationModels.GRAD.value:
                continue
            
            title = "{} with {}".format(classifier, wv_model)
            predictions = load_test_result(classifier, wv_model)["Predictions"]
            show_confusion_matrix(test_labels, predictions, title)


