import ConfigParser
import csv
import json
import operator
import os
import re
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ate

config = ConfigParser.ConfigParser()
config.readfp(open(os.path.join('ate/configuration', 'config.ini')))

min_term_length = int(config.get('main', 'min_term_length'))
min_term_words = int(config.get('main', 'min_term_words'))
stopwords = json.loads(config.get('main', 'stopwords'))
term_patterns=json.loads(config.get('main', 'term_patterns'))

source_dataset=config.get('main', 'source_dataset')

out_file_cval=config.get('main', 'out_file_cval')
out_file_tf = config.get('main', 'out_file_tf')


def c_values(terms, trace=False):
    terms_df = pd.DataFrame(terms, columns=['term'])
    terms_df['w'] = 1
    terms_df['len'] = len(terms_df['term'])
    term_stats = terms_df.groupby(['term'])['w'].agg([np.sum])
    term_stats['len'] = list(pd.Series(term_stats.index).apply(lambda x:len(x)))
    term_stats.sort_values(by=['len'], ascending=True, inplace=True)
    term_series = list(term_stats.index)
    vectorizer = CountVectorizer(analyzer='word')
    vectorizer.fit(term_series)
    term_vectors = vectorizer.transform(term_series)
    n_terms = len(term_series)

    # significant speadup
    is_part_of = [(ti, term_series[j], 1) for i, ti in enumerate(term_vectors) for j, tj in enumerate(term_vectors) if cosine_similarity(ti, tj) and tj.find(ti) >= 0]

    subterms = pd.DataFrame(is_part_of, columns=['term', 'part_of', 'w']).set_index(['term', 'part_of'])
    c_values = []
    for t in term_series:
        # print t
        current_term = term_stats.loc[t]
        # average frequency of the superterms
        c_value = 0
        if t in subterms.index:
            subterm_of = list(subterms.loc[t].index)
            for st in subterm_of:
                c_value -= term_stats.loc[st]['sum']
            c_value /= float(len(subterm_of))
        # add current term frequency
        c_value += current_term['sum']
        # multiply to log(term length)
        c_value = c_value * np.log(current_term['len'])
        #if trace:
        	#print t, 'freq=', current_term['sum'], ' cvalue=', c_value
        c_values.append(c_value)
        # break
    return sorted(zip(term_series, c_values), key=lambda x: x[1], reverse=True)


def terms_freq(terms = [], use_cached_terms=True):
    if use_cached_terms:
        with open('data/terms/temp/c_val_raw_terms.txt', 'r') as tfile:
            terms = tfile.read().splitlines()

    tf = Counter(terms)
    return reversed(sorted(tf.items(), key=operator.itemgetter(1)))



def extract_terms(source_file = source_dataset, method = 'cval', out_file = out_file_cval):
    with open(source_file, "r") as sf:
        dataset = sf.read()

    dataset = re.sub(r'et +al\.', 'et al', dataset)
    dataset = unicode(dataset, "utf-8", errors='ignore')
    dataset = re.split(r'[\r\n]', dataset)

    term_extractor = ate.TermExtractor(stopwords=stopwords, term_patterns=term_patterns, min_term_words=min_term_words,
                                       min_term_length=min_term_length)
    if method == 'cval':
        terms = term_extractor.extract_terms(dataset)
        terms_freqs = term_extractor.c_values(terms, trace=False)
        with open(out_file, 'wb') as csvfile:
            termwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
            for cv in terms_freqs:
                termwriter.writerow(cv)
    else:
        terms_freqs = terms_freq()
        with open(out_file, 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in terms_freqs:
                writer.writerow([key, value])

    return terms_freqs



if __name__ == '__main__':
    terms = extract_terms(method='tf', out_file=out_file_tf)