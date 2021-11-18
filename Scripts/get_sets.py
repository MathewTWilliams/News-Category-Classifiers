
from constants import TRAIN_SET_PATH, VALID_SET_PATH, TEST_SET_PATH
from save_load_json import load_json



def get_training_set(): 
    return get_set(TRAIN_SET_PATH)

def get_validation_set(): 
    return get_set(VALID_SET_PATH)


def get_test_set(): 
    return get_set(TEST_SET_PATH)




def get_set(set):
    set_dict = load_json(set)
    set_bags = []
    set_labels = []
    for category, article_list in set_dict.items(): 
        set_labels.extend([category for i in range(len(article_list))])
        set_bags.append(article_list)
    
    return set_bags, set_labels
