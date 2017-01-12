import spacy
from dateutil import parser as dp
import random

# English tokenizer, tagger, parser, NER, etc
nlp = spacy.load('en')


def ner_extract(text, ents_type=None):
    """generic func to extract location and time"""
    output = list()
    for i in nlp(text).ents:
        if ents_type and i.label_ in ents_type:
            output.append((i.label_, i.text))
        elif not ents_type:
            output.append((i.label_, i.text))
    return output


def str_to_datetime(text):
    """using fuzzy convert to change time to datetime object"""
    concat = ' '.join([i[1] for i in text])
    return dp.parse(concat, fuzzy=True)


string =
'''
Ticket Details:
Wifredo Lam exhibition
at Exhibitions at Tate Modern, London
Ticket: A103
1 Adult ticket allocated
Exhibition at £14.50 each on
Sunday 27th November at 14:00
'''

date_time = ner_extract(string, ents_type=['DATE', 'TIME'])
location = ner_extract(string, ents_type=['GPE'])[0]
org = ner_extract(string, ents_type=['ORG'])[0]

msg_template = ['Your invitation at {0}, {1} is confirmed at {2}.',
                'We at {0} are thrilled to have your visit at {1}, {2}']


def msg_gen(repo):
    n = random.randint(0, len(repo) - 1)
    msg = repo[n].format(org[1], location[1], str_to_datetime(date_time))
    return msg

if __name__ == '__main__':
    print(msg_gen(msg_template))
