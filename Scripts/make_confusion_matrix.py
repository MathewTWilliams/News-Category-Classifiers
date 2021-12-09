#Author: Matt Williams
#Version: 12/06/2021
#Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
    plt.show()


