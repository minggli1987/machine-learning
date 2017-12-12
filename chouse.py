#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chouse

script calling UK Companies House API for company profile data.
A rate limit of 600 requests per 5 minutes imposed.
"""

import os
import sys
import base64
from pprint import pprint

import requests

# pylint: disable=C0111

BASE_URL = "https://api.companieshouse.gov.uk/"
DOC_URL = "https://document-api.companieshouse.gov.uk/document/{0}/content"
API_KEY = os.getenv("CHKEY")
TOKEN = "Basic " + base64.b64encode((API_KEY + ":").encode('ascii')).decode('ascii')
SESS = requests.Session()
SESS.headers.update({"Authorization": TOKEN, "Accept": "*/*"})
VALS = ' '.join(sys.argv[1:])


def search_company(query, default_s=SESS, url=BASE_URL+"search/companies"):
    resp = default_s.get(url, params={"q": query, "items_per_page": 5})
    print(resp.request.headers)

    if resp.ok:
        return resp.json()


def check_profile(registered_id, default_s=SESS, url=BASE_URL+"/company/{0}"):
    resp = default_s.get(url.format(registered_id))
    print(resp.request.headers)

    if resp.ok:
        return resp.json()


def detail_profile(json, key='filing_history', default_s=SESS, url=BASE_URL):
    try:
        suburl = json["links"][key]
    except KeyError:
        raise Exception("requested detail doesn't exist")
    resp = default_s.get(url + suburl)
    print(resp.request.headers)

    if resp.ok:
        return resp.json()


def fetch_document(unique_identifier, default_s=SESS, url=DOC_URL):
    resp = default_s.get(url.format(unique_identifier))
    print(resp.request.headers)

    with open('./doc.pdf', 'wb') as filename:
        filename.write(resp.content)
    return resp.status_code


if __name__ == "__main__":

    if not API_KEY:
        raise Exception("missing Companies House API Key")
    elif not VALS:
        raise Exception("missing search phase or company id.")

    if VALS.isdigit():
        base_profile = check_profile(VALS)
        pprint(base_profile)
        pprint(detail_profile(base_profile))
    elif ' ' in VALS or len(VALS) < 20:
        pprint(search_company(VALS))
    else:
        print(fetch_document(VALS))
