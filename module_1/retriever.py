import nltk
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.similarities import Similarity
from gensim.matutils import cossim
from gensim import utils


class Retriever(object):

    def __init__(self, BASE_PATH):
        self.dictionary = Dictionary.load_from_text(
            f'{BASE_PATH}/data/dictionary.txt.bz2')
        self.tfidf_model = TfidfModel.load(
            f'{BASE_PATH}/data/tfidf.tfidf_model')
        self.id_to_titles = utils.unpickle(
            f'{BASE_PATH}/data/bow.mm.metadata.cpickle')
        self.index = Similarity.load(f'{BASE_PATH}data/index.mm')

    def getTfidfForText(self, text):
        text = text.replace('\n', ' ')
        tokens = nltk.word_tokenize(text.lower())
        bow_vec = self.dictionary.doc2bow(tokens)
        return self.tfidf_model[bow_vec]

    def findSimilarToVector(self, input_tfidf, topn=5):
        sims = self.index[[input_tfidf]][0]
        sims = sorted(sims, key=lambda item: -item[1])
        results = sims[0:topn]
        results = [(self.id_to_titles[id], score) for id, score in results]
        return results

    def get_similar(self, query):
        query_tfidf = self.getTfidfForText(query)
        results = self.findSimilarToVector(query_tfidf)
        return results
