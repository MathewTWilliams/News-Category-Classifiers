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
- **classifier_metrics.py**: contains a method to calculate different classification metrics on the results of a classifier's predictions and save those metrics to a JSON file. Also has a method to find the best result for a given classifier and word vector model based on a specified metric. 

-  **cross_validation.py**: contains a method to run [Halving Grid Search Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html) on a given classifier and parameter grid. 

- **get_article_vectors.py**: contains methods for accessing the mean word embedding for each scrapped article in the training and test sets. 

- **run_classification.py**: contains a method for running the given classifier on the test set and saving the results of the predictions to a JSON file. 

- **save_load_json.py**: contains methods for loading and saving json files. 

- **get_vec_models.py**: contains methods to load and return the downloaded word vector models. 

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
8. **cross_validation_util.py**: runs Halving Grid Search cross validation on all of our classifiers. Each classifier has a parameter grid defined in their respective files. These can be edited for greater hyper-parameter tuning. 
## License
Distributed under the MIT License. See ```LICENSE.txt``` for more information. 

## Contact
- Mathew Williams
- email: williams.mathew.t@gmail.com
- Project Link: https://github.com/MathewTWilliams/News-Category-Classifiers 
