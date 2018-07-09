import ConfigParser
import os
import itertools

import numpy as np

config = ConfigParser.ConfigParser()
config.readfp(open(os.path.join('ate/configuration', 'config.ini')))

source_terms='../data/terms/temp/c_val_raw_terms.txt'

out_file_groups=config.get('main', 'out_file_cval')

class Seeder:
    def __init__(self):
        self.seeds = set()
        self.cache = dict()

    def get_seed(self, word):
        LIMIT = 4
        seed = self.cache.get(word,None)
        if seed is not None:
            return seed
        for seed in self.seeds:
            if self.distance(seed, word) <= LIMIT:
                self.cache[word] = seed
                return seed
        self.seeds.add(word)
        self.cache[word] = word
        return word

    def distance(self, s1, s2):
        l1 = len(s1)
        l2 = len(s2)
        matrix = [range(zz,zz + l1 + 1) for zz in xrange(l2 + 1)]
        for zz in xrange(0,l2):
            for sz in xrange(0,l1):
                if s1[sz] == s2[zz]:
                    matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz])
                else:
                    matrix[zz+1][sz+1] = min(matrix[zz+1][sz] + 1, matrix[zz][sz+1] + 1, matrix[zz][sz] + 1)
        return matrix[l2][l1]

from pyjarowinkler import distance


def jaro_winkler(terms):
    def get_similarity(el):
        first, second = el
        return (first, second), distance.get_jaro_distance(first, second, winkler=True, scaling=0.1)
    term_pairs = itertools.combinations(np.unique(terms), 2)
    similarities = map(get_similarity, term_pairs)
    return list(similarities)


def group_similar(terms):
    seeder = Seeder()
    terms = sorted(terms, key=seeder.get_seed)
    groups = itertools.groupby(terms, key=seeder.get_seed)
    return [list(v) for k,v in groups]


if __name__ == '__main__':
    with open(source_terms, 'r') as tfile:
        terms = tfile.read().splitlines()
    groups  = group_similar(np.unique(terms[:300]))
    jw_similarities = jaro_winkler(terms[:300])
    print()


