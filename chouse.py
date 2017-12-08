#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chouse

script calling company house API for company profile data.
A rate limit of 600 requests per 5 minutes imposed.
"""

import os
import sys
import requests

# pylint: disable=C0111

BASE_URL = "https://api.companieshouse.gov.uk/"
API_KEY = os.environ["CHKEY"]
SESS = requests.Session()
SESS.auth = (API_KEY, "")
VALS = ' '.join(sys.argv[1:])


def search_company(query, default_s=SESS, url=BASE_URL+"search/companies"):
    resp = default_s.get(url, params={"q": query, "items_per_page": 5})
    if resp.status_code == 200:
        return resp.json()


def search_profile(registered_id, default_s=SESS, url=BASE_URL+"/company/{0}"):
    resp = default_s.get(url.format(registered_id))
    if resp.status_code == 200:
        return resp.json()


if __name__ == "__main__":

    if not API_KEY:
        raise Exception("missing Company House API Key")
    elif not VALS:
        raise Exception("missing search phase or company id.")

    if VALS.isdigit():
        print(search_profile(VALS))
    else:
        print(search_company(VALS))
