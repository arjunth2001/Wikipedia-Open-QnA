import json
import logging
import os

import torch
import torch.nn.functional as F

from tqdm import tqdm
import drqa
import utils
from dictionary import Dictionary
from dataset import ReadingDataset, BatchSampler
from dotdict import DotDictify
args = {
    "seed": 42,
    "data": "/scratch/arjunth2001/data",
    "max_tokens": 16000,
    "batch_size": 32,
    "num_workers": 4,
    "max_epoch": 100,
    "clip_norm": 10,
    "lr": 2e-3,
    "momentum": 0.99,
    "weight_decay": 0.0,
    "lr_shrink": 0.1,
    "min_lr": 1e-6,
    "log_file": "/scratch/arjunth2001/logs/train.log",
    "tune_embed": 1000,
    "checkpoint_dir": "./models",
    'embed_dim': 300,
    'embed_path': '/scratch/arjunth2001/data/glove.840B.300d.txt',
    'hidden_size': 128,
    'context_layers': 3,
    'question_layers': 3,
    'dropout': 0.4,
    'bidirectional': True,
    'concat_layers': True,
    'question_embed': True,
    'use_in_question': True,
    'use_lemma': True,
    'use_pos': True,
    'use_ner': True,
    'use_tf': True,

}
args = DotDictify(args)


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('device: ' + str(device))

    torch.manual_seed(args.seed)

    # Load a dictionary
    dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
    print(
        'Loaded a word dictionary with {} words'.format(len(dictionary)))

    # Load a training and validation dataset
    with open(os.path.join(args.data, 'train.json')) as file:
        train_contents = json.load(file)
        train_dataset = ReadingDataset(
            args, train_contents['contexts'], train_contents['examples'], dictionary, skip_no_answer=True, single_answer=True)

    with open(os.path.join(args.data, 'dev.json')) as file:
        contents = json.load(file)
        valid_dataset = ReadingDataset(
            args, contents['contexts'], contents['examples'], dictionary, feature_dict=train_dataset.feature_dict, skip_no_answer=True, single_answer=True
        )

    # Build a model
    model = drqa.DrQA.build_model(args, dictionary).to(device)
    print('Built a model with {} parameters'.format(
        sum(p.numel() for p in model.parameters())))

    # Build an optimizer and a learning rate schedule
    optimizer = torch.optim.Adamax(
        model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=args.lr_shrink)

    # Load last checkpoint if one exists
    utils.load_checkpoint(args, model, optimizer, lr_scheduler, device)

    for epoch in range(args.max_epoch):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=args.num_workers, collate_fn=train_dataset.collater,
            batch_sampler=BatchSampler(
                train_dataset, args.max_tokens, args.batch_size, shuffle=True, seed=args.seed)
        )
        model.train()
        stats = {'loss': 0., 'lr': 0., 'num_tokens': 0.,
                 'batch_size': 0., 'grad_norm': 0., 'clip': 0.}
        progress_bar = tqdm(
            train_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

        for batch_id, sample in enumerate(progress_bar):
            # Forward and backward pass
            # if batch_id == 10:
            # break
            sample = utils.move_to_device(sample, device)
            start_scores, end_scores = model(
                sample['context_tokens'], sample['question_tokens'],
                context_features=sample['context_features']
            )
            start_loss = F.nll_loss(start_scores, torch.LongTensor(
                sample['answer_start']).view(-1).to(device))
            end_loss = F.nll_loss(end_scores, torch.LongTensor(
                sample['answer_end']).view(-1).to(device))
            loss = start_loss + end_loss
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients and fix embeddings of infrequent words
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_norm)
            if args.tune_embed is not None and hasattr(model, 'embedding'):
                model.embedding.weight.grad[args.tune_embed:] = 0
            optimizer.step()

            # Update statistics for progress bar
            stats['loss'] += loss.item()
            stats['lr'] += optimizer.param_groups[0]['lr']
            stats['num_tokens'] += sample['num_tokens'] / len(sample['id'])
            stats['batch_size'] += len(sample['id'])
            stats['grad_norm'] += grad_norm
            stats['clip'] += 1 if grad_norm > args.clip_norm else 0
            progress_bar.set_postfix({key: '{:.4g}'.format(
                value / (batch_id + 1)) for key, value in stats.items()}, refresh=True)

        print('Epoch {:03d}: {}'.format(epoch, ' | '.join(
            key + ' {:.4g}'.format(value / len(progress_bar)) for key, value in stats.items())))

        # Adjust learning rate based on validation result
        f1_score = validate(args, model, valid_dataset, epoch, device)
        lr_scheduler.step(f1_score)
        utils.save_checkpoint(args, model, optimizer,
                              lr_scheduler, epoch, f1_score)
        if optimizer.param_groups[0]['lr'] <= args.min_lr:
            logging.info('Done training!')
            break


def validate(args, model, valid_dataset, epoch, device):
    model.eval()
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, num_workers=args.num_workers, collate_fn=valid_dataset.collater,
        batch_sampler=BatchSampler(
            valid_dataset, args.max_tokens, args.batch_size, shuffle=True, seed=args.seed)
    )

    stats = {'start_acc': 0., 'end_acc': 0., 'token_match': 0.,
             'f1': 0., 'exact_match': 0., 'num_tokens': 0., 'batch_size': 0.}
    progress_bar = tqdm(
        valid_loader, desc='| Epoch {:03d}'.format(epoch), leave=False)

    for batch_id, sample in enumerate(progress_bar):
        sample = utils.move_to_device(sample, device)
        with torch.no_grad():
            start_scores, end_scores = model(
                sample['context_tokens'], sample['question_tokens'],
                context_features=sample['context_features']
            )
            start_target, end_target = sample['answer_start'], sample['answer_end']

            stats['num_tokens'] += sample['num_tokens']
            stats['batch_size'] += len(sample['id'])

            start_pred, end_pred, _ = model.decode(
                start_scores, end_scores, max_len=15)
            stats['start_acc'] += sum(ex_pred in ex_target for ex_pred,
                                      ex_target in zip(start_pred, start_target))
            stats['end_acc'] += sum(ex_pred in ex_target for ex_pred,
                                    ex_target in zip(end_pred, end_target))

            for i, (start_ex, end_ex) in enumerate(zip(start_pred, end_pred)):
                # Check if the pair of predicted tokens in the targets
                stats['token_match'] += any((start_ex == s and end_ex == t)
                                            for s, t in zip(start_target[i], end_target[i]))

                # Official evaluation
                text_target = valid_dataset.answer_texts[sample['id'][i]]
                context = valid_dataset.contexts[valid_dataset.context_ids[sample['id'][i]]]
                #print("hola", start_ex, end_ex)
                start_idx = context['offsets'][start_ex][0]
                end_idx = context['offsets'][end_ex][1]
                text_pred = context['text'][start_idx: end_idx]
                stats['exact_match'] += utils.metric_max_over_ground_truths(
                    utils.exact_match_score, text_pred, text_target)
                stats['f1'] += utils.metric_max_over_ground_truths(
                    utils.f1_score, text_pred, text_target)

        progress_bar.set_postfix({key: '{:.3g}'.format(value / (stats['batch_size'] if key != 'batch_size' else (batch_id + 1)))
                                  for key, value in stats.items()}, refresh=True)

    print('Epoch {:03d}: {}'.format(epoch, ' | '.join(key + ' {:.3g}'.format(
        value / (stats['batch_size'] if key != 'batch_size' else len(progress_bar))) for key, value in stats.items())))

    return stats['f1'] / stats['batch_size']


if __name__ == '__main__':
    main()
