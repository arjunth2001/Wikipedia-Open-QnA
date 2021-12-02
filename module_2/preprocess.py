import itertools
import json
import os

from multiprocessing import Pool
from tokenizer import SpacyTokenizer
from dictionary import Dictionary
from dotdict import DotDictify
from tqdm.auto import tqdm
TOK = None
# Each process has its own tokenizer


def init_tokenizer(tokenizer, annotators):
    global TOK
    TOK = SpacyTokenizer(annotators=annotators)

# Multiprocessing requires global function


def tokenize(text):
    global TOK
    return TOK.tokenize(text)


def tokenize_all(texts, tokenizer, annotators, num_workers=None):
    """Tokenization might take a long time, even when done in parallel"""
    workers = Pool(num_workers, init_tokenizer,
                   initargs=[tokenizer, annotators])
    tokens = workers.map(tokenize, texts)
    workers.close()
    workers.join()
    return tokens


args = {
    "data": "/scratch/arjunth2001/data/squad/",
    "dest_dir": "/scratch/arjunth2001/data/",
    "num_workers": 12,
    "threshold": 5,
    "num_words": -1,
    "embed_path": "/scratch/arjunth2001/data/glove.840B.300d.txt",
    "restrict-vocab": False,
}
args = DotDictify(args)


def main():
    os.makedirs(args.dest_dir, exist_ok=True)
    for split in ['train', 'dev']:
        # Load JSON dataset
        dataset_path = os.path.join(args.data, '{}-v1.1.json'.format(split))
        dataset = load_dataset(dataset_path)
        print('Loaded dataset {} ({} questions, {} contexts)'.format(
            dataset_path, len(dataset['questions']), len(dataset['contexts'])))

        # Tokenize questions and contexts and build a dictionary
        questions = tokenize_all(dataset['questions'], args.tokenizer, [
                                 'lemma'], args.num_workers)
        contexts = tokenize_all(dataset['contexts'], args.tokenizer, [
                                'lemma', 'pos', 'ner'], args.num_workers)

        # Build a dictionary from train examples only
        if split == 'train':
            build_word_dictionary(args, [q['tokens'] for q in questions], [
                                  c['tokens'] for c in contexts])

        examples = []
        for qid, cid in tqdm(enumerate(dataset['context_ids'])):
            answer_spans, answer_texts = [], []
            for ans in dataset['answers'][qid]:
                # Map answers to token spans
                char_start, char_end = ans['answer_start'], ans['answer_start'] + len(
                    ans['text'])
                token_start = [i for i, tok in enumerate(
                    contexts[cid]['offsets']) if tok[0] == char_start]
                token_end = [i for i, tok in enumerate(
                    contexts[cid]['offsets']) if tok[1] == char_end]

                # Bad tokenization can lead to no answer found
                if len(token_start) == 1 and len(token_end) == 1:
                    answer_spans.append((token_start[0], token_end[0]))
                    answer_texts.append(ans['text'])

            examples.append({
                'id': dataset['question_ids'][qid],
                'answers': {'spans': answer_spans, 'texts': answer_texts},
                'question': {key: questions[qid][key] for key in ['tokens', 'lemma']},
                'context_id': cid,
            })

        # Write preprocessed data to file
        output = {'contexts': contexts, 'examples': examples}
        output_file = os.path.join(args.dest_dir, '%s.json' % split)
        with open(output_file, 'w') as file:
            json.dump(output, file)
            print('Wrote {} examples to {}'.format(
                len(examples), output_file))


def load_dataset(filename):
    """Parse a JSON file into a dictionary"""
    with open(filename, 'r') as file:
        data = json.load(file)['data']
    outputs = {'question_ids': [], 'questions': [],
               'answers': [], 'contexts': [], 'context_ids': []}
    for article in data:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                outputs['question_ids'].append(qa['id'])
                outputs['questions'].append(qa['question'])
                outputs['answers'].append(qa['answers'])
                outputs['context_ids'].append(len(outputs['contexts']))
            outputs['contexts'].append(paragraph['context'])
    return outputs


def build_word_dictionary(args, question_tokens, context_tokens):
    """Build a word dictionary from questions and contexts"""
    valid_words = None
    if args.restrict_vocab and args.embed_path is not None:
        with open(args.embed_path) as file:
            valid_words = {SpacyTokenizer.normalize(
                line.rstrip().split(' ')[0]) for line in file}

    dictionary = Dictionary()
    for text in itertools.chain(question_tokens, context_tokens):
        for word in text:
            if valid_words is None or word in valid_words:
                dictionary.add_word(word)

    dictionary.finalize(threshold=args.threshold, num_words=args.num_words)
    dictionary.save(os.path.join(args.dest_dir, 'dict.txt'))
    print('Built a dictionary with {} words'.format(len(dictionary)))
    return dictionary


if __name__ == '__main__':
    main()
