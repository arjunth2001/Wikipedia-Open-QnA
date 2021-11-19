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

    # Download this file to get the latest wikipedia dump:
    # https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    # Rename it as enwiki.xml.bz2 and put it inside data folder.
    # dump_file = f'{BASEPATH}/data/wiki-mini.bz2'
    dump_file = f'{BASEPATH}/data/enwiki.xml.bz2'

    # ======== STEP 1: Build Dictionary =========
    dictionary = Dictionary()
    wiki = WikiCorpus(dump_file, dictionary=dictionary)
    print('Parsing Wikipedia to build Dictionary...')
    sys.stdout.flush()
    t0 = time.time()
    dictionary.add_documents(wiki.get_texts(), prune_at=None)
    print('    Building dictionary took %s' % formatTime(time.time() - t0))
    print('    %d unique tokens before pruning.' % len(dictionary))
    sys.stdout.flush()
    keep_words = 100000
    wiki.dictionary.filter_extremes(
        no_below=20, no_above=0.1, keep_n=keep_words)
    wiki.dictionary.save_as_text(f'{BASEPATH}/data/dictionary.txt.bz2')

    # ======== STEP 2: Convert Articles To Bag-of-words ========
    # Now that we have our finalized dictionary, we can create bag-of-words
    # representations for the Wikipedia articles. This means taking another
    # pass over the Wikipedia dump!
    dictionary = Dictionary.load_from_text(
        f'{BASEPATH}/data/dictionary.txt.bz2')
    wiki = WikiCorpus(dump_file, dictionary=dictionary)

    # Turn on metadata so that wiki.get_texts() returns the article titles.
    wiki.metadata = True

    print('\nConverting to bag of words...')
    sys.stdout.flush()
    t0 = time.time()

    # Generate bag-of-words vectors (term-document frequency matrix) and
    # write these directly to disk.
    # By setting metadata = True, this will also record all of the article
    # titles into a separate pickle file, 'bow.mm.metadata.cpickle'
    MmCorpus.serialize(f'{BASEPATH}/data/bow.mm', wiki,
                       metadata=True, progress_cnt=10000)

    print('    Conversion to bag-of-words took %s' %
          formatTime(time.time() - t0))
    sys.stdout.flush()
    # Load the article titles back
    id_to_titles = utils.unpickle(f'{BASEPATH}/data/bow.mm.metadata.cpickle')

    # Create the reverse mapping, from article title to index.
    titles_to_id = {}

    # For each article...
    for at in id_to_titles.items():
        # `at` is (index, (pageid, article_title))  e.g., (0, ('12', 'Anarchism'))
        # at[1][1] is the article title.
        # The pagied property is unused.
        titles_to_id[at[1][1]] = at[0]

    # Store the resulting map.
    utils.pickle(titles_to_id, f'{BASEPATH}/data/titles_to_id.pickle')

    # We're done with the article titles so free up their memory.
    del id_to_titles
    del titles_to_id

    # To clean up some memory, we can delete our original dictionary and
    # wiki objects, and load back the dictionary directly from the file.
    del dictionary
    del wiki
    dictionary = Dictionary.load_from_text(
        f'{BASEPATH}//data/dictionary.txt.bz2')

    # Load the bag-of-words vectors back from disk.
    corpus_bow = MmCorpus(f'{BASEPATH}/data/bow.mm')

    # ======== STEP 3: Learn tf-idf model ========
    # At this point, we're all done with the original Wikipedia text, and we
    # just have our bag-of-words representation.
    # Now we can look at the word frequencies and document frequencies to
    # build a tf-idf model which we'll use in the next step.
    print('\nLearning tf-idf model from data...')
    t0 = time.time()
    # Build a Tfidf Model from the bag-of-words dataset.
    model_tfidf = TfidfModel(
        corpus_bow, id2word=dictionary, normalize=False)
    print('    Building tf-idf model took %s' %
          formatTime(time.time() - t0))
    model_tfidf.save(f'{BASEPATH}/data/tfidf.tfidf_model')

    # ======== STEP 4: Convert articles to tf-idf ========
    # We've learned the word statistics and built a tf-idf model, now it's time
    # to apply it and convert the vectors to the tf-idf representation.
    print('\nApplying tf-idf model to all vectors...')
    t0 = time.time()

    # Apply the tf-idf model to all of the vectors.
    # This took 1hr. and 40min. on my machine.
    # The resulting corpus file is large--17.9 GB for me.
    MmCorpus.serialize(f'{BASEPATH}/data/corpus_tfidf.mm',
                       model_tfidf[corpus_bow], progress_cnt=10000)
    print('    Applying tf-idf model took %s' %
          formatTime(time.time() - t0))
    print("Done!")
