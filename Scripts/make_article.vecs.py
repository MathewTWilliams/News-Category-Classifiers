import pandas as pd
from save_load_json import load_json
from constants import TRAIN_SET_PATH,CATEGORIES, get_article_vec_path
from multiprocessing import Pool
from get_vec_models import get_vec_models



def make_article_vec(category, article_list, model_wv, model_num, model_name): 
    cat_dfs = pd.DataFrame()
    num_articles = len(article_list)

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
        cat_dfs.append(article_vec, ignore_index=True)
        print(category, "articles finished:", i + 1, "/", num_articles, "model_num:", model_num)

    article_df.insert(len(article_df.columns), "Category", [category for _ in range(num_articles)], False)
    name = model_name + "_" + category + "_" + "article_vecs.json"
    cat_dfs.to_json(get_article_vec_path(name))
    




def make_article_vectors(model, model_name, model_num):  


    train_dict = load_json(TRAIN_SET_PATH)

    with Pool(processes=len(CATEGORIES)) as pool:
        inputs = zip(CATEGORIES, \
                    [train_dict[category] for category in CATEGORIES], \
                    [model for _ in range(len(CATEGORIES))], \
                    [model_num for _ in range(len(CATEGORIES))], \
                    [model_name for _ in range(len(CATEGORIES))])
        pool.starmap(make_article_vec, inputs)

   


if __name__ == "__main__": 
    model_dict = get_vec_models()

    count = 1
    for name, model in model_dict.items(): 
        make_article_vectors(model, name, count)
        count += 1



    
    





    
    
    

