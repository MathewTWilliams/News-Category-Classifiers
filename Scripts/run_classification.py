#Author: Matt Williams
#Version: 6/03/2022


from get_article_vectors import get_training_info, get_test_info
from classifier_metrics import calculate_classifier_metrics
from make_confusion_matrix import show_confusion_matrix

def run_classifier(vec_model_name, classifier, classifier_details):
    
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    classifier.fit(training_data, training_labels)
    predictions = classifier.predict(test_data)

    calculate_classifier_metrics(test_labels, predictions, classifier_details)
    title = "{} w/{} Confusion Matrix".format(classifier_details['Model'], vec_model_name)
    show_confusion_matrix(test_labels, predictions)
