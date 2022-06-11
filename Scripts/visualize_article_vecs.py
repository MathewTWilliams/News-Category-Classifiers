#Author: Matt Williams
#Version: 11/27/2021

#Reference: https://towardsdatascience.com/google-news-and-leo-tolstoy-visualizing-word2vec-word-embeddings-with-t-sne-11558d8bd4d
from get_article_vectors import get_combined_train_test_info
from get_vec_models import get_vec_model_names
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import RAND_STATE



color_dict = {
    "MEDIA":"red", 
    "WEIRD NEWS":"yellow",
    "GREEN":"green",
    "WORLDPOST":"purple",
    "RELIGION":"pink", 
    "STYLE":"magenta",
    "SCIENCE":"blue",
    "WORLD NEWS":"black",
    "TASTE":"orange",
    "TECH":"gray",
    #for clustering algorithms that label with integers
    #key value is based on index in CATEGORIES list 
    0:"red", 
    1:"yellow",
    2:"green",
    3:"purple",
    4:"pink", 
    5:"magenta",
    6:"blue",
    7:"black",
    8:"orange",
    9:"gray"

    
} 

   


def visualize_article_vecs(vec_model_name, n_components, labels = []): 
    '''Given a vector model name, the number of axes to project to, and the\ prediction labels (optional), 
    use T-SNE to visualized the article vectors with the given labels. If labels is left as an empty list, 
    then we are just visualizing the article vectors and their real labels'''

    if n_components < 2 or n_components > 3: 
        return

    if  len(labels) == 0: 
        data, labels = get_combined_train_test_info(vec_model_name)
    else: 
        data, _ = get_combined_train_test_info(vec_model_name)

    tsne = TSNE(n_components=n_components, init="pca", random_state=RAND_STATE)


    new_values = tsne.fit_transform(data)

    fig = plt.figure(figsize = (25,25))
    if n_components == 3: 
        ax = fig.add_subplot(111, projection ='3d')
    else: 
        ax = fig.add_subplot(111)
    for i,value in enumerate(new_values): 
        if n_components == 3:
            x = value[0]
            y = value[1]
            z = value[2]
            ax.scatter(x,y,z, color=color_dict[labels.iat[i]])
        elif n_components == 2: 
            x = value[0]
            y = value[1]
            ax.scatter(x,y, color=color_dict[labels.iat[i]])

    plt.show()



if __name__ == "__main__": 
    for vec_model_name in get_vec_model_names():
        visualize_article_vecs(vec_model_name, 2)
        visualize_article_vecs(vec_model_name, 3)