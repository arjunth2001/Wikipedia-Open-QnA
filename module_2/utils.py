import collections
import os
import logging
import re
import string
import sys
import torch
from torch._C import device

from torch.serialization import default_restore_location
from tokenizer import SpacyTokenizer


def load_embedding(embed_weight, embed_path, dictionary):
    """Parse an embedding text file into an torch.nn.Embedding layer."""
    word_counts = {}
    with open(embed_path) as file:
        for line in file:
            tokens = line.rstrip().split(" ")
            assert(len(tokens) == embed_weight.size(1) + 1)
            word = SpacyTokenizer.normalize(tokens[0])
            if word in dictionary.word2idx:
                vector = torch.Tensor([float(weight) for weight in tokens[1:]])
                if word not in word_counts:
                    word_counts[word] = 1
                    embed_weight[dictionary.index(word)].copy_(vector)
                else:
                    word_counts[word] = word_counts[word] + 1
                    embed_weight[dictionary.index(word)].add_(vector)

    for word, count in word_counts.items():
        embed_weight[dictionary.index(word)].div_(count)
    logging.info('Loaded {} / {} word embeddings ({:.2f}%)'.format(len(word_counts),
                 len(dictionary), 100 * len(word_counts) / len(dictionary)))


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch, f1_score):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_epoch = getattr(save_checkpoint, 'last_epoch', -1)
    save_checkpoint.last_epoch = max(last_epoch, epoch)
    prev_best = getattr(save_checkpoint, 'best_score', float('-inf'))
    save_checkpoint.best_score = max(prev_best, f1_score)

    state_dict = {
        'epoch': epoch,
        'f1_score': f1_score,
        'best_score': save_checkpoint.best_score,
        'last_epoch': save_checkpoint.last_epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
    }
    if f1_score > prev_best:
        torch.save(state_dict, os.path.join(
            args.checkpoint_dir, 'model.pt'))


def load_checkpoint(args, model, optimizer, lr_scheduler, device):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.pt")
    if os.path.isfile(checkpoint_path):
        state_dict = torch.load(
            checkpoint_path, map_location=device)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        save_checkpoint.best_score = state_dict['best_score']
        save_checkpoint.last_epoch = state_dict['last_epoch']
        return state_dict


def move_to_device(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device)
    elif isinstance(sample, list):
        return [move_to_device(x, device) for x in sample]
    elif isinstance(sample, dict):
        return {key: move_to_device(value, device) for key, value in sample.items()}
    else:
        return sample


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(
        prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """Check if the prediction is a (soft) exact match with the ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def regex_match_score(prediction, pattern):
    """Check if the prediction matches the given regular expression."""
    try:
        compiled = re.compile(
            pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        logging.warn('Regular expression failed to compile: %s' % pattern)
        return False
    return compiled.match(prediction) is not None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Given a prediction and multiple valid answers, return the score of
    the best prediction-answer_n pair given a metric function.
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
