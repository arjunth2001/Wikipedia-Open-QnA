import spacy
import unicodedata


class SpacyTokenizer(object):
    """Based on https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/spacy_tokenizer.py"""

    def __init__(self, model='en_core_web_sm', annotators=['lemma', 'pos', 'ner']):
        self.annotators = annotators
        disabled_components = ['parser']
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            disabled_components.append('tagger')
        if 'ner' not in self.annotators:
            disabled_components.append('ner')
        self.nlp = spacy.load(model, disable=disabled_components)

    def tokenize(self, text):
        tokens = self.nlp(text.rstrip())
        return {
            'text': text,
            'tokens': [self.normalize(t.text) for t in tokens],
            'offsets': [(t.idx, t.idx + len(t)) for t in tokens],
            'pos': [t.tag_ for t in tokens] if 'pos' in self.annotators else None,
            'lemma': [t.lemma_ for t in tokens] if 'lemma' in self.annotators else None,
            'ner': [t.ent_type_ for t in tokens] if 'ner' in self.annotators else None,
        }

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)
