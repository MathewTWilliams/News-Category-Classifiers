#Author: Matt Williams
#Version: 11/21/2021


from sklearn.metrics import accuracy_score, multilabel_confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

from save_load_json import save_json
from constants import get_result_path_with_name




def calculate_classifier_metrics(true_labels, predictions, model_details): 
    '''Given the true_labels, the predictions of the classifer, and the classifier details, 
    calculate the primary metrics for classification and save them to a json file along with
    the model details.'''
    model_details['Accuracy'] = accuracy_score(true_labels, predictions)
    #model_details['Confusion_Matrix'] = multilabel_confusion_matrix(true_labels, predictions)
    model_details['F1_Score'] = f1_score(true_labels, predictions, average="macro")
    model_details['Precision'] = precision_score(true_labels, predictions, average="macro")
    model_details['Recall'] = recall_score(true_labels, predictions, average="macro")
    #model_details['ROC_AUC'] = roc_auc_score(true_labels, predictions)


    file_path = get_result_path_with_name(model_details['Classifier'])
    save_json(model_details, file_path)
