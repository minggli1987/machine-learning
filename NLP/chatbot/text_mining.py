import requests
from bs4 import BeautifulSoup
import re
import time
import pandas as pd

web_pages = {
    1: 'http://www.nhs.uk/Conditions/Heart-block/Pages/Symptoms.aspx',
    2: 'http://www.nhs.uk/conditions/frozen-shoulder/Pages/Symptoms.aspx',
    3: 'http://www.nhs.uk/conditions/coronary-heart-disease/'
    'Pages/Symptoms.aspx',
    4: 'http://www.nhs.uk/conditions/bronchitis/Pages/Symptoms-old.aspx',
    5: 'http://www.nhs.uk/conditions/warts/Pages/Introduction.aspx',
    6: 'http://www.nhs.uk/conditions/Sleep-paralysis/Pages/Introduction.aspx',
    7: 'http://www.nhs.uk/Conditions/Glue-ear/Pages/Symptoms.aspx',
    8: 'http://www.nhs.uk/Conditions/Depression/Pages/Symptoms.aspx',
    9: 'http://www.nhs.uk/Conditions/Turners-syndrome/Pages/Symptoms.aspx',
    10: 'http://www.nhs.uk/Conditions/Obsessive-compulsive-disorder/'
    'Pages/Symptoms.aspx'
}

#
# for i in range(1, len(web_pages) + 1, 1):
#
#     m = re.search('conditions/(.*)/pages/', web_pages[i].lower()).group(1)
#     m = re.sub('[^0-9a-zA-Z]+', ' ', m)
#     web_pages[m] = web_pages.pop(i)
#
# illness = list(web_pages.keys())
# illness.sort(reverse=True)
# print(illness)

for i in list(web_pages.keys()):
    r = requests.get(url=web_pages[i])
    soup = BeautifulSoup(r.text, 'html5lib')
    # for i in soup.find_all('h1'):
    #     print(i)

# web page overview
print('scrapping web page for...{0}.'.format(soup.title.string), '\n\n')
time.sleep(2)
html = soup.prettify()
# print(html, flush=False)

desc_attributes = {
    'name': 'description'
}

subj_attributes = {
    'name': 'DC.Subject',
    'scheme': 'NHSC.Ontology'

}

article_attributes = {
    'start_t_0': 'Overview',
    'start_t_1': '\n        Print this page\n    ',
    'start_t_2': '\n     \n    \n        Print this page\n    \n    \n     \n\n',
    'end_t_0': 'Share:',
    'end_t_1': '',
    'end_t_2': ''
}

article = list()

for i in soup.find_all(['p', 'li', 'meta']):
    article.append(i.get_text())

start_t_2 = article.index(article_attributes['start_t_2'])
print(article[start_t_2:])