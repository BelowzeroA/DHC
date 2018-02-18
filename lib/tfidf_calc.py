import collections
import math
import os

from lib.text_processor import TextProcessor


class TfidfCalculator:

    def __init__(self, text_processor: TextProcessor, trash_words):
        self.text_processor = text_processor
        self.corpus = []
        self.term_frequency = []
        self.collection = []
        self.terms = []
        self.tfidf = []
        self.trash_words = trash_words


    def load_corpus(self, collection):
        self.corpus = self.text_processor.extract_words_from_list(collection)


    def calculate_tf(self, terms):
        term_frequency = collections.Counter(terms)
        term_frequency = {x: term_frequency[x] for x in term_frequency if term_frequency[x] > 3}

        for i in term_frequency:
            term_frequency[i] = term_frequency[i] / float(len(terms))

        return term_frequency


    def calculate_idf(self, word):
        encounters = self.corpus.count(word)
        if encounters == 0:
            encounters = 1
        return math.log(len(self.corpus) / encounters)


    def calculate_tfidf(self, target_collection):
        tfidf = []
        terms_collection = self.text_processor.extract_words_from_list(target_collection)
        term_frequency = self.calculate_tf(terms_collection)
        for term in term_frequency:
            if term in self.trash_words:
                continue
            idf = self.calculate_idf(term)
            tfidf.append((term, term_frequency[term] * idf))
        tfidf.sort(key=lambda tup: tup[1], reverse=True)
        self.tfidf = dict(tfidf)
        return self.tfidf


    def get_term_tfidf(self, term):
        if term in self.tfidf:
            return self.tfidf[term]
        return None

    def get_top_terms(self, top=0):
        if top:
            return list(self.tfidf)[0:top]
        else:
            return list(self.tfidf)


    # def save_frequency_dictionary(self, filename):
    #     with open(filename, mode='wt', encoding='utf-8') as output_file:
    #         for term, freq in self.term_frequency.most_common():
    #             print(term, freq, file=output_file)

    def save_tfidf_dictionary(self, filename):
        with open(filename, mode='wt', encoding='utf-8') as output_file:
            for term in self.tfidf:
                print(term, self.tfidf[term], file=output_file)


    def load_tfidf_dictionary(self, filename):
        if not os.path.isfile(filename):
            print('TF-IDF dictionary failed to load from', filename)
            return False
        self.tfidf = {}
        with open(filename, mode='r', encoding='utf-8') as output_file:
            for line in output_file:
                parts = line.split()
                self.tfidf[parts[0]] = float(parts[1])
        return True