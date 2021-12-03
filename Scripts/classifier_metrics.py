#Author: Matt Williams
#Version: 12/02/2021

from sklearn.metrics import classification_report, hamming_loss, jaccard_score  
from save_load_json import save_json
from constants import make_result_path
from numpy import NINF, PINF
from constants import get_result_path
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


def find_best_result(classifier_name, vec_model_name, metric, large = True): 
    '''Given the classifier name, the vector model used, and the name of the classifier metric, 
    return the file name that contains the best metric value for the given parameters'''
    best_score = NINF if large else PINF
    best_file_name = ""
    for file in os.listdir(get_result_path(classifier_name,"")):
        if file.startswith(vec_model_name):
            results = load_json(get_result_path(classifier_name, file))['Classification_Report']
            if metric not in results.keys(): 
                return ""
            if large and results[metric] > best_score:
                best_file_name = file
                best_score = results[metric] 
            elif not large and results[metric] < best_score: 
                best_file_name = file
                best_score = results[metric]
    
    return best_file_name
