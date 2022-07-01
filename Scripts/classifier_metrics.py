#Author: Matt Williams
#Version: 12/08/2021

from sklearn.metrics import classification_report, hamming_loss, jaccard_score  
from save_load_json import save_json
from utils import make_result_path
from numpy import NINF, PINF
from utils import get_result_path
from save_load_json import load_json
import os 




def calculate_classifier_metrics(true_labels, predictions, model_details): 
    '''Given the true_labels, the predictions of the classifer, and the classifier details, 
    calculate a classification report and save them to a json file along with
    the model details.'''
    model_details['Classification_Report'] = classification_report(true_labels, predictions, output_dict=True, 
                                                                    digits=4)

    model_details['Hamming_Lose'] = hamming_loss(true_labels, predictions)
    model_details['Jaccard_Score'] = jaccard_score(true_labels, predictions, average="macro")
    model_details['Predictions'] = predictions.tolist()

    file_path = make_result_path(model_details['Model'], model_details['Vector_Model'])
    save_json(model_details, file_path)

