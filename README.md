# News-Category-Classifiers
A repository for my for graduate Data Mining I semester project. 

## Problem Description
Given the words from an article's headline, description, and body, predict which sugject/category the article belongs to. 

## Dataset
The dataset used for this project can be found at the link below. 

- [News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)

The dataset contain information (the category, the link, etc.) about ~200k news articles from HuffPost, which were collected between 2012 and 2018.

The original dataset contained 41 categories, for this project I used a subset of 10 categories. I choose the 10 categories that would theoretically give use the most balanced subset. 

The 10 categories chosen were: media, weird news, green, worldpost, religion, style, science, world news, taste, and tech. 

Once all the articles for each category were scrapped for the web, the number of articles for each category decreased due to the fact that some of the article weren't available anymore. Below you can find the category, the number of articles used, and the number of articles in the data set. 

- Media: 2505 / 2815
- Weird News: 2372 / 2670
- Green: 2284 / 2622
- WorldPost: 2549 / 2579
- Religion: 2455 / 2556
- Style: 2232 / 2254
- Science: 2016 / 2178
- World News: 2172 / 2177
- Taste: 2068 / 2096
- Tech: 1773 / 2082

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

## Getting Started

### Prerequisites
- A newer version of Python must be installed. This was made using Python 3.7.9
- edit: project has been update for Python 3.10.2

### Installation
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

## Usage
### Modules: 

- **ada_boost.py**: contains a method to run the Ada Boost algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **bagging.py**: contains a method to run the Bagging classification algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **classifier_metrics.py**: contains a method to calculate different classification metrics on the results of a classifier's predictions and save those metrics to a JSON file. Also has a method to find the best result for a given classifier and word vector model based on a specified metric. 

-  **cross_validation.py**: contains a method to run [Halving Grid Search Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html) on a given classifier and parameter grid. 

- **get_article_vectors.py**: contains methods for accessing the mean word embedding for each scrapped article in the training and test sets. 

- **get_vec_models.py**: contains various methods to access downloaded gensim word vector models. 

- **grad_boost.py**: contains a method to run the Gradient Boosting classification algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **knn.py**: contains a method to run the K-Nearest Neighbor algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **logistic-regression.py**: contains a method to run the Logistic Regression algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **mlp.py**: contains a method to run the Multi-Layer Perceptron algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **naive_bayes.py**: contains a method to run the Complement Naive Bayes algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **near_centroid.py**: contains a method to run the Nearest Centroid algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **near_radius.py**: contains a method to run the Radius Nearest Neighbors algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation.  

- **random_forest.py**: contains a method to run the Random Forest algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **run_classification.py**: contains a method for running a given classifier on the test set of a given word vector model and saving the results of the predictions to a JSON file. 

- **save_load_json.py**: contains various methods for loading and saving json files.

- **svm.py**: contains a method to run the Support Vector Machine algorithm on the test set of a given word vector model. Also contains a parameter grid for cross validation. 

- **text_cleaner.py**: contains a method for cleaning a given string of text. The file contains the following capabilities:

  - Removing digits
  - Converting numbers to words
  - Removing single characters
  - Removing stop words
  - Removing special characters
  - Lemmatization
  - Stemming
  - Expanding contractions
  - Converting Uppercase to lower case letters
  - Removing a given number of the most and least frequent words in the given text

- **utils.py**: contains global variables, enumeration definitions, and provides file path related methods. 
### Order of Scripts

1. **sort_dataset.py**: the original data set contains ~200k json objects. This script will group those JSON objects in a dictionary based on the category that each article object belongs to. That dictionary is then saved to a new JSON file. 

2. **make_article_set.py**: using the new JSON object from the previous script, we make another JSON file that contains the important information for each article of the categories we have chosen for this project. An article isn't added to the set if its article no longer exists. This article set is saved to a new JSON file.  

3. **web_scraper.py**: using multi-threading, scrape the text for each article found in the JSON file made by the previous script. A new JSON file is created to store the scraped text, where they are stored in a dictionary based on their respective categories. 

4. **re-scrape.py**: this script tries to re-scrape articles that weren't successfully scraped by the previous script. This script doesn't use multi-threading in order to prevent too many requests from being sent in a short period of time, which is the most likely reason why articles weren't correctly scrapped the first time.

5. **clean_article_texts.py**: this script utilizes multi-processing and the text_cleaner.py module in order to clean the scrapped texts. The cleaned texts are saved to a new JSON file where the texts are grouped by category. The value of the dictionary is a 3 dimensional list. 

6. **download_vec_models.py**: a simple script to download and save the word vector models I've chosen for this project via the Gensim library.

7. **make_article_vecs.py**: using the cleaned text of the articles, the mean word vector for each article is calculated and saved to new JSON files. This is down for each word vector model on both the training and test sets. 

8. **visualize_article_vecs.py**: after making our word vectors for each article, we can use this script in order to visualize the word vectors for each word vector model using feature reduction techniques like t-SNE along with PCA. 

9. **cross_validation_util.py**: runs Halving Grid Search cross validation on all of our classifiers, on all word vector models. Each classifier has a parameter grid defined in their respective files. These can be edited for greater hyper-parameter tuning. 

10. **run_all_classifiers.py**: after cross validation has been completed, the best parameter combinations for each classifier on each word vector model have been saved to JSON files. Using these best parameter combinations, run each classifier on each vector model on the test set. Results are then saved to JSON files. 

11. **make_bar_graphs.py**: visualize the accuracies of the results. Only used accuracies here because the results showed very similar numbers across precision, recall, and f1-score. Bar graphs are made for each classifier and each word vector model. 

12. **make_confusion_matrix.py**: visualize how well each classifier on each word vector model did on the test set. 

## Results

| Classifier        | Word Vector Model | Accuracy |
| ----------------- | ----------------- | -------- |
| Support Vector Machine | FastText          | 81.7%    |
| Multi-Layer Perceptron | Word2Vec          | 80.9%    |
| Support Vector Machine | Word2Vec          | 80.7%    |
| Multi-Layer Perceptron | FastText          | 80.6%    |
| Logistic Regression    | Word2Vec          | 80.2%    |
| Multi-Layer Perceptron | GLoVe             | 80.0%    |
| Logistic Regression    | GLoVe             | 79.9%    |
| Logistic Regression    | FastText          | 79.7%    |
| Support Vector Machine | GLoVe             | 79.5%    |
| K-Nearest Neighbor     | GLoVe             | 76.5%    |
| Bagging                | FastText          | 76.2%    |
| Bagging                | GLoVe             | 76.1%    |
| K-Nearest Neighbor     | Word2Vec          | 76.1%    |
| Bagging                | Word2Vec          | 76.0%    |
| K-Nearest Neighbor     | FastText          | 74.8%    |
| Nearest Centroid       | Word2Vec          | 69.8%    |
| Nearest Centroid       | GLoVe             | 68.8%    |
| Random Forest          | GLoVe             | 67.1%    |
| Nearest Centroid       | FastText          | 66.5%    |
| Random Forest          | Word2Vec          | 66.0%    |
| Random Forest          | FastText          | 64.3%    |
| Ada-Boost              | GLoVe             | 62.8%    |
| Ada-Boost              | FastText          | 62.1%    | 
| Comp. Naive Bayes      | Word2Vec          | 62.1%    |
| Comp. Naive Bayes      | GLoVe             | 60.8%    |
| Ada-Boost              | Word2Vec          | 60.7%    |
| Comp. Naive Bayes      | FastText          | 51.7%    |


## License
Distributed under the MIT License. See ```LICENSE.txt``` for more information. 

## Contact
- Mathew Williams
- email: williams.mathew.t@gmail.com
- Project Link: https://github.com/MathewTWilliams/News-Category-Classifiers 
