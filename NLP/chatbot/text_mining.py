import requests
from bs4 import BeautifulSoup
import re

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


for i in range(1, len(web_pages) + 1, 1):

    m = re.search('conditions/(.*)/pages/', web_pages[i].lower()).group(1)
    m = re.sub('[^0-9a-zA-Z]+', ' ', m)
    web_pages[m] = web_pages.pop(i)

illness = list(web_pages.keys())
i = illness[1]
r = requests.get(url=web_pages[i])
data = r.text
soup = BeautifulSoup(data, 'lxml')

# html = soup.prettify()
print(soup.title.string)


for link in soup.find_all(['p']):
    txt = link.string
    print(txt)


