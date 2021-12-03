# News-Category-Classifiers-and-Clusterings
A repository for my Project for Data Mining I and Concepts in Artifical Intelligence. My goal is to build a few different news category classifiers and compare them. In addition to this, I want to implement a few clustering algorithms and see how well the different algorithms can cluster the same articles. Both the Classifiers and the Clustering algorithms will be tested on multiple dense word vector representation (Word2Vec, GLoVe, and FastText). 


## Built With: 
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Contractions](https://github.com/kootenpv/contractions)
- [Gensim](https://radimrehurek.com/gensim/)
- [Matplotlib](https://matplotlib.org/)
- [NLTK](https://www.nltk.org/)
- [num2words](https://github.com/savoirfairelinux/num2words)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pyspellchecker](https://github.com/barrust/pyspellchecker)
- [requests](https://docs.python-requests.org/en/latest/)
- [Sci-Kit Learn](https://scikit-learn.org/stable/)

# Getting Started

## Prerequisites
- A newer version of Python must be installed. This was made using Python 3.7.9

## Installation
1. Clone the repository: 
```sh
git clone https://github.com/MathewTWilliams/News-Category-Classifiers
```
2. Make a new virtual environment
```sh
python -m venv '/path/to/env'
```
3. Activate Virtual Environment (on Windows)
```sh
./path/to/env/Scripts/activate
```

4. Install requirements
```sh
pip install -r /path/to/requirements.txt
```
5. Download the data set: https://www.kaggle.com/rmisra/news-category-dataset

6. Make an folder Named 'Data' that sits in the same location as the Scripts folder from the repository.

7. Move the downloaded dataset (keeping the same name) into the newly made 'Data' Folder.

# Usage
## Scripts 
Here is a general overview of each script contained in the project. Each file contains comments that give more information on each script. 

- sort_dataset.py: sort our dataset based on categories and saves to a JSON file. 

- make_article_set.py: form our subset of the dataset to be used for the project. Grabs articles of the selected categories and makes sure the articles are still accessible. Set is saved to a JSON file. 

- web_scraper.py: Scrape the article text for each article that was selected by 'make_article_set.py'. Scraped text is saved to a JSON file. 

- text_cleaner.py: contains our methods to clean the text for the articles. Returns a a 2d-list where each row is a sentence and each column is a word in a sentence. 

- clean_article_texts.py: Take all of the scrapped text and clean it using our 'text_cleaner.py' script.

- download_vec_models.py: Download the selected pre-trained Word Vector Models and save them. 

- get_vec_models.py: script used to access the downloaded Word Vector Models. 

- make_article_vecs.py: take our cleaned text for each article and calculate an embedding for each article. After doing so for each article, make our training and test data set splits. 

- get_article_vectors.py: Gives access to our article vectors. Can obtain either the training data set, test data, or both (combined). 


## Utility Scripts
- constant.py: contains constant values important to the project. Mostly contains file path related items. 

- save_load_json.py: contains methods to load and save json files when not using Pandas related items. 

- re-scrape.py: The full explanation of this script is found at the top of the script itself. Needed to rescrape certain urls that weren't successfully scraped from web_scraper.py due to overloading the HuffPost server with too many requests. 

- classifier_metrics.py: calculates and saves a classification report given true labels, predictions, and the vector model used. 

- clustering_metrics.py: calculates and saves a report of import clustering metrics given clustering results and true labels. Since true labels are known, a classification report is also done. 

- visualize_article_vecs.py: used to visualizing plotting the article vectors and the results of the clustering algorithms. 

- cross_validation.py: Used for K-fold cross validation for classifiers. 

## Sci-kit Learn related scripts:
- Contains the implementations of the different ML related algorithms used. 
  - k-means.py
  - hierarchical_clustering.py
  - db_scan.py
  - spectral_clustering.py
  - random_forest.py
  - ml_perceptron.py
  - naive_bayes.py
  - knn.py
  - svm.py
  - logistic_regression.py

# License
Distributed under the MIT License. See ```LICENSE.txt``` for more information. 

# Contact
- Mathew Williams
- email: williams.mathew.t@gmail.com
- Project Link: https://github.com/MathewTWilliams/News-Category-Classifiers 
