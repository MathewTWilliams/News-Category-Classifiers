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
### Custom Python Modules: 
- **classifier_metrics.py**: contains a method to calculate different classification metrics on the results of a classifier's predictions and save those metrics to a JSON file. Also has a method to find the best result for a given classifier and word vector model based on a specified metric. 
-  **cross_validation.py**: contains a method to run [Halving Grid Search Cross Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html) on a given classifier and parameter grid. 

### Order of Scripts

## License
Distributed under the MIT License. See ```LICENSE.txt``` for more information. 

## Contact
- Mathew Williams
- email: williams.mathew.t@gmail.com
- Project Link: https://github.com/MathewTWilliams/News-Category-Classifiers 
