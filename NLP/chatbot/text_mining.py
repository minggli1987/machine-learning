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


r = requests.get(url=web_pages[1])
data = r.text
soup = BeautifulSoup(data, 'html5lib')

# html = soup.prettify()
print(soup.title.string)


for link in soup.find_all(['li']):
    txt = link.string
    if txt and len(txt.replace(' ', '')) > 5:
        print(txt, end='\n\n')


