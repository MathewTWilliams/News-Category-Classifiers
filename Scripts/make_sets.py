from sklearn.model_selection import train_test_split
from save_load_json import load_json, save_json
from constants import *

def make_sets(): 
    """Take out cleaned text data and split it into our training, 
    test, and validation sets."""
    cleaned_texts = load_json(CLEANED_TEXT_PATH) 
    train_dict = {}
    test_dict = {}
    val_dict = {}
    for category in CATEGORIES: 
        article_list = cleaned_texts[category]

        train, val_test = train_test_split(article_list,
                                            train_size=TRAIN_SET_PERC, 
                                            random_state=TTS_RAND_STATE)
        train_dict[category] = train

        sum_perc = TRAIN_SET_PERC + VALID_SET_PERC
        valid, test = train_test_split(val_test, 
                                        test_size= (sum_perc/2), 
                                        random_state=TTS_RAND_STATE)

        test_dict[category] = test
        val_dict[category] = valid


    save_json(train_dict, TRAIN_SET_PATH)
    save_json(val_dict, VALID_SET_PATH)
    save_json(test_dict, TEST_SET_PATH)


if __name__ == "__main__":
    make_sets()