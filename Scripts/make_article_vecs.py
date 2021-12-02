import pandas as pd
from save_load_json import load_json
from constants import *
from multiprocessing import Pool
from get_vec_models import *
from sklearn.model_selection import train_test_split
import os

#Reference: https://www.kaggle.com/ananyabioinfo/text-classification-using-word2vec 
def make_article_vecs(category, article_list, model_wv, model_num, model_name): 
    '''given a category, article_list, and a model's word vectors, 
    calculate the average word vector for each article. Then split the 
    vectors into training and test splits and save them to a json file.
    model_num and model_name are purely for debugging purposes.'''
    cat_df = pd.DataFrame()
    num_articles = len(article_list)

    #calculate average vector for each article in a category
    for (i,article) in enumerate(article_list): 
        article_df = pd.DataFrame()
        for sentence in article: 
            for word in sentence: 
                try:
                    word_vec = model_wv[word]
                    article_df= article_df.append(pd.Series(word_vec), ignore_index=True)
                except:
                    continue
        
        
        article_vec = article_df.mean()
        cat_df = cat_df.append(pd.Series(article_vec), ignore_index=True)
        print(category, "articles finished:", i + 1, "/", num_articles, "model_num:", model_num)

    # Insert labels
    cat_df.insert(len(cat_df.columns), \
                        "Category", \
                        [category for _ in range(len(cat_df.index))], \
                         False)

    # Make our train and test split
    train_cat_df, test_cat_df = train_test_split(cat_df, \
                                                        train_size=TRAIN_SET_PERC, \
                                                        test_size=TEST_SET_PERC, \
                                                        random_state= RAND_STATE)
    #Save each set to a different folder
    name = model_name + "_" + category + "_" + "article_vecs.json"
    train_cat_df.to_json(get_article_vecs_path("Train" , name))
    test_cat_df.to_json(get_article_vecs_path("Test" , name))

def fill_article_nan_values(subfolder, category, model_name= ""): 
    '''Given a subfolder(Train or Test), a category, and a model_name load in the article vectors
    associated with those values. replace any NaN values found with the median value of the NaN value
    is found in. '''

    for file in os.listdir(get_article_vecs_path(subfolder,"")): 
        model_check = True if model_name == "" else file.startswith(model_name)
        if model_check and file.find(category) != -1: 
            path = get_article_vecs_path(subfolder, file)
            article_df = pd.read_json(path)
            #find our columns that have NaN values
            bool_vector = article_df.isnull().any()
            #-1 so we don't touch the category column
            for i in range(len(article_df.columns) - 1): 
                if bool_vector[i]: 
                    col_median = article_df[str(i)].median()
                    article_df[str(i)].fillna(value=col_median, inplace=True)
            article_df.to_json(path)
            
                   

def combine_categroy_article_vecs(model_name): 
    '''Given a model name, this method goes through the Test and 
    Train folders to combine all data frames that used the model and
    combine them into a single dataframe.  '''
    #combine all training data frames related to the given model
    train_combined_df = pd.DataFrame()
    for file in os.listdir(get_article_vecs_path("Train", "")): 
        if file.startswith(model_name): 
            category_df_path = get_article_vecs_path("Train", file)
            category_df = pd.read_json(category_df_path)
            train_combined_df = train_combined_df.append(category_df, ignore_index=True)
           
    file_name = model_name + "_training_set.json"
    train_combined_df.to_json(get_article_vecs_path("Train", file_name))

    #combine all test data frames related to the given model
    test_combined_df = pd.DataFrame()
    for file in os.listdir(get_article_vecs_path("Test", "")): 
        if file.startswith(model_name): 
            category_df_path = get_article_vecs_path("Test", file)
            category_df = pd.read_json(category_df_path)
            test_combined_df = test_combined_df.append(category_df, ignore_index=True)
           

    file_name = model_name + "_test_set.json"
    test_combined_df.to_json(get_article_vecs_path("Test", file_name))

if __name__ == "__main__": 
    #train_dict = load_json(CLEANED_TEXT_PATH)

    #this is done so all 3 models aren't loaded into memory at once
    #name, model_wv = get_w2v_model()
    #for category in CATEGORIES: 
    #    make_article_vecs(category, train_dict[category], model_wv, 1, name)

    #name, model_wv = get_fasttext_model()
    #for category in CATEGORIES: 
    #    make_article_vecs(category, train_dict[category], model_wv, 2, name)

    #name, model_wv = get_glove_model()
    #for category in CATEGORIES: 
    #   make_article_vecs(category, train_dict[category], model_wv, 3, name)

    fill_article_nan_values("Train", "GREEN")
    for name in get_model_names(): 
        combine_categroy_article_vecs(name)
        

    



    
    





    
    
    

