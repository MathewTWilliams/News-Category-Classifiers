#Author: Matt Williams
#Version: 6/24/2022


from get_article_vectors import get_training_info, get_test_info
from classifier_metrics import calculate_classifier_metrics
from make_confusion_matrix import show_confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import minmax_scale

def run_classifier(vec_model_name, classifier, classifier_details):
    '''Given the name of the word vector model, a sklearn classifier instance, and a dictionary of the classifier details,
    load the dataset and run the classification algorithm. The results are then saved to a JSON file. '''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)


    if type(classifier) is ComplementNB: 
        training_data = minmax_scale(training_data, feature_range=(0,1))
        test_data = minmax_scale(test_data, feature_range=(0,1))


    classifier.fit(training_data, training_labels)
    predictions = classifier.predict(test_data)

    calculate_classifier_metrics(test_labels, predictions, classifier_details)
    title = "{} w/{} Confusion Matrix".format(classifier_details['Model'], vec_model_name)
    show_confusion_matrix(test_labels, predictions)
