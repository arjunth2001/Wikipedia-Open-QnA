# -*- coding: utf-8 -*-

"""
Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.
This script was built on the one provided in gensim:
`gensim.scripts.make_wikicorpus`
"""

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, WikiCorpus, MmCorpus
from gensim import similarities
from gensim import utils
import time
import sys
import logging
import os

BASEPATH = "/scratch/arjunth2001"


def formatTime(seconds):
    """
    Takes a number of elapsed seconds and returns a string in the format h:mm.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d" % (h, m)


if __name__ == '__main__':
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Logging
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(filename=f'{BASEPATH}/log.txt',
                        format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%H:%M:%S')
    logging.root.setLevel(level=logging.INFO)
    print("Building Index...")
    t0 = time.time()
    dictionary = Dictionary.load_from_text(
        f'{BASEPATH}/data/dictionary.txt.bz2')
    model_tfidf = TfidfModel.load(f'{BASEPATH}/data/tfidf.tfidf_model')
    corpus_bow = MmCorpus(f'{BASEPATH}/data/bow.mm')
    index = similarities.Similarity(
        output_prefix=f"{BASEPATH}/data/tfidf_", corpus=model_tfidf[corpus_bow], num_features=100000, num_best=5)
    index.save(f"{BASEPATH}/data/index.mm")
    print('    Building Index took %s' %
          formatTime(time.time() - t0))
    print("Done!")
