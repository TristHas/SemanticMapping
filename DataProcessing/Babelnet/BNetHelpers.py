#!/usr/bin/env python
# -*- coding: utf-8 -*-
import requests
import urllib2
import urllib
import json
import gzip
from StringIO import StringIO
import pandas as pd

class BNetSparqlClient():
    def __init__(self):
        self.bnet_key = "0d8faefb-181c-41d4-b094-f06afd7e3d95"
        self.endpoint = "http://babelnet.org/sparql/"
        self.encoding = {
            "?":"%3F",
            " ":"+",
            ":":"%3A",
            "/":"%2F",
            "#":"%23"
        }
        self.formats = {
            "html": "&format=text%2Fhtml",
            "csv" : "&format=text%2Fcsv"
        }

    def query_url(self, q, f = "csv"):
        for char in self.encoding:
            q = q.replace(char, self.encoding[char])
        return "".join([
                self.endpoint,
                "?query=",
                q,
                self.formats[f]
            ])

    def query(self, q, f = "csv"):
        res = requests.get(self.query_url(q, f))
        if res.status_code == 200:
            return pd.read_csv(StringIO(res.content))
        else:
            raise Exception("Request failed")




class BNetHttpClient():
    def __init__(self):
        """
            DOC
        """
        self.service_url = 'https://babelnet.io/v3/'
        self.key  = "d862aa17-089c-4505-be35-7a75689177f0"

    def get_syns(self, id):
        """
            DOC
        """
        service = "getSynset"
        params = {
            'key':self.key,
            'id' :id
        }
        url = self.service_url + service + '?' + urllib.urlencode(params)
        request = urllib2.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib2.urlopen(request)
        if response.info().get('Content-Encoding') == 'gzip':
            buf = StringIO( response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            return data
        else:
            print "returning response"
            return response

    def get_edges(self, id):
        service = "getEdges"
        params = {
            'key':self.key,
            'id' :id
        }
        url = self.service_url + service + '?' + urllib.urlencode(params)
        request = urllib2.Request(url)
        request.add_header('Accept-encoding', 'gzip')
        response = urllib2.urlopen(request)
        if response.info().get('Content-Encoding') == 'gzip':
            buf = StringIO( response.read())
            f = gzip.GzipFile(fileobj=buf)
            data = json.loads(f.read())
            return data
        else:
            print "returning response"
            return response

    def treat_data(self, data):
        """
            DOC
        """
        senses = data['senses']
        for result in senses:
                lemma = result.get('lemma')
                language = result.get('language')
                print language.encode('utf-8') + "\t" + str(lemma.encode('utf-8'))

        print '\n'
        # retrieving BabelGloss data
        glosses = data['glosses']
        for result in glosses:
                gloss = result.get('gloss')
                language = result.get('language')
                print language.encode('utf-8') + "\t" + str(gloss.encode('utf-8'))

        print '\n'
        # retrieving BabelImage data
        images = data['images']
        for result in images:
                url = result.get('url')
                language = result.get('language')
                name = result.get('name')
                print language.encode('utf-8') +"\t"+ str(name.encode('utf-8')) +"\t"+ str(url.encode('utf-8'))
