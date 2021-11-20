import pandas as pd
from save_load_json import load_json
from constants import CATEGORIES, get_article_vecs_path, CLEANED_TEXT_PATH
from constants import TEST_SET_PERC, TRAIN_SET_PERC, VALID_SET_PERC, TTS_RAND_STATE
from multiprocessing import Pool
from get_vec_models import get_vec_models
from sklearn.model_selection import train_test_split



def make_article_vecs(category, article_list, model_wv, model_num, model_name): 
    cat_df = pd.DataFrame()
    num_articles = len(article_list)

    #calculate average vector for each article
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
        cat_df.append(article_vec, ignore_index=True)
        print(category, "articles finished:", i + 1, "/", num_articles, "model_num:", model_num)

    # Insert labels
    cat_df.insert(len(cat_df.columns), \
                        "Category", \
                        [category for _ in range(num_articles)], \
                         False)

    # Make our train, valid, test split
    train_cat_df, valid_test_cat_df = train_test_split(cat_df, \
                                                        test_size=TRAIN_SET_PERC, \
                                                        random_state= TTS_RAND_STATE)
                                                        
    valid_test_perc = (VALID_SET_PERC + TEST_SET_PERC) / 2
    valid_cat_df, test_cat_df = train_test_split(valid_test_cat_df, \
                                            test_size= valid_test_perc, \
                                            random_state=TTS_RAND_STATE)

    #Save each set to a different folder
    name = model_name + "_" + category + "_" + "article_vecs.json"
    train_cat_df.to_json(get_article_vecs_path("Train/" + name))
    valid_cat_df.to_json(get_article_vecs_path("Valid/" + name))
    test_cat_df.to_json(get_article_vecs_path("Test/" + name))



def make_category_article_vectors(model, model_name, model_num):  


    train_dict = load_json(CLEANED_TEXT_PATH)

    with Pool(processes=len(CATEGORIES)) as pool:
        inputs = zip(CATEGORIES, \
                    [train_dict[category] for category in CATEGORIES], \
                    [model for _ in range(len(CATEGORIES))], \
                    [model_num for _ in range(len(CATEGORIES))], \
                    [model_name for _ in range(len(CATEGORIES))])
        pool.starmap(make_article_vecs, inputs)

   


if __name__ == "__main__": 
    model_dict = get_vec_models()

    count = 1
    for name, model in model_dict.items(): 
        make_category_article_vectors(model, name, count)
        count += 1



    
    





    
    
    

