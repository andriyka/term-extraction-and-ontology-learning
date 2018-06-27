import ConfigParser
import ate
import csv
import re
import json

config = ConfigParser.ConfigParser()
config.readfp(open('config.ini'))

min_term_length = int(config.get('main', 'min_term_length'))
min_term_words = int(config.get('main', 'min_term_words'))
stopwords = json.loads(config.get('main', 'stopwords'))
term_patterns=json.loads(config.get('main', 'term_patterns'))
doc_file=config.get('main', 'doc_file')
out_file=config.get('main', 'out_file')

fp = open(doc_file, "r")
doc_txt = fp.read() 
fp.close()
doc_txt = unicode(doc_txt, "utf-8", errors='ignore')
doc_txt = re.sub(r'et +al\.', 'et al', doc_txt)
doc_txt = re.split(r'[\r\n]', doc_txt)


def tf_idf(terms):
    print terms

term_extractor = ate.TermExtractor(stopwords=stopwords, term_patterns=term_patterns, min_term_words=min_term_words, min_term_length=min_term_length)
terms = term_extractor.extract_terms(doc_txt)
c_values = term_extractor.c_values(terms, trace=True)

with open(out_file, 'wb') as csvfile:
    termwriter = csv.writer(csvfile, delimiter=';', quotechar='', quoting=csv.QUOTE_NONE)
    for cv in c_values:
        termwriter.writerow(cv)


