import numpy as np
import pandas as pd
from settings import setting
from text_mining import NHSTextMining
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews as mr
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from sklearn import model_selection


__author__ = 'Ming Li'

web_pages = {
    0: 'http://www.nhs.uk/Conditions/Heart-block/Pages/Symptoms.aspx',
    1: 'http://www.nhs.uk/conditions/frozen-shoulder/Pages/Symptoms.aspx',
    2: 'http://www.nhs.uk/conditions/coronary-heart-disease/'
    'Pages/Symptoms.aspx',
    3: 'http://www.nhs.uk/conditions/bronchitis/Pages/Symptoms-old.aspx',
    4: 'http://www.nhs.uk/conditions/warts/Pages/Introduction.aspx',
    5: 'http://www.nhs.uk/conditions/Sleep-paralysis/Pages/Introduction.aspx',
    6: 'http://www.nhs.uk/Conditions/Glue-ear/Pages/Symptoms.aspx',
    7: 'http://www.nhs.uk/Conditions/Depression/Pages/Symptoms.aspx',
    8: 'http://www.nhs.uk/Conditions/Turners-syndrome/Pages/Symptoms.aspx',
    9: 'http://www.nhs.uk/Conditions/Obsessive-compulsive-disorder/'
    'Pages/Symptoms.aspx'
}

urls = list(web_pages.values())

web_scraper = NHSTextMining(urls=urls, attrs=setting, n=None, display=True)
data = web_scraper.extract()

# first is subject, second is description of the pages, rest is main article

for i in range(1, 10):
    print(data[web_pages[i]][:1])
    input('press enter to continue...')

heart_block = data[web_pages[0]]
cleansed_string = NHSTextMining.cleanse(heart_block)

print(cleansed_string[0])

# regressand = word_tokenize(cleansed_string[0])
# regressors = word_tokenize(cleansed_string[1:])



question = '''
i feel depressed
'''

print(word_tokenize(question))

# train

# model_selection.train_test_split()