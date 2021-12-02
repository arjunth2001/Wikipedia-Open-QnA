import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from layers import StackedRNN, BilinearAttention, SequenceAttention, SelfAttention


class DrQA(nn.Module):
    def __init__(
        self, dictionary, embed_dim=300, hidden_size=128, context_layers=3, question_layers=3, dropout=0.4,
        bidirectional=True, concat_layers=True, question_embed=True, pretrained_embed=None, num_features=0,
    ):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = dropout
        self.question_embed = question_embed
        self.num_features = num_features
        self.embedding = nn.Embedding(
            len(dictionary), embed_dim, dictionary.pad_idx)
        if pretrained_embed is not None:
            print("Loading Embedding..")
            utils.load_embedding(self.embedding.weight.data,
                                 pretrained_embed, dictionary)

        if question_embed:
            self.context_question_attention = SequenceAttention(embed_dim)

        self.context_rnn = StackedRNN(
            input_size=(1 + question_embed) * embed_dim + num_features, hidden_size=hidden_size,
            num_layers=context_layers, dropout=dropout, bidirectional=bidirectional, concat_layers=concat_layers
        )

        self.question_rnn = StackedRNN(
            input_size=embed_dim, hidden_size=hidden_size, num_layers=question_layers,
            dropout=dropout, bidirectional=bidirectional, concat_layers=concat_layers
        )

        context_size = question_size = (1 + bidirectional) * hidden_size
        if concat_layers:
            context_size = context_size * context_layers
            question_size = question_size * question_layers

        self.question_attention = SelfAttention(question_size)
        self.start_attention = BilinearAttention(question_size, context_size)
        self.end_attention = BilinearAttention(question_size, context_size)

    @classmethod
    def build_model(cls, args, dictionary):
        return cls(
            dictionary, embed_dim=args.embed_dim, hidden_size=args.hidden_size, context_layers=args.context_layers,
            question_layers=args.question_layers, dropout=args.dropout, bidirectional=args.bidirectional,
            question_embed=args.question_embed, concat_layers=args.concat_layers,
            pretrained_embed=args.embed_path, num_features=args.num_features,
        )

    def forward(self, context_tokens, question_tokens, **kwargs):
        # Ignore padding when calculating attentions
        context_mask = torch.eq(context_tokens, self.dictionary.pad_idx)
        question_mask = torch.eq(question_tokens, self.dictionary.pad_idx)

        # Embed context and question words
        context_embeddings = self.embedding(context_tokens)
        question_embeddings = self.embedding(question_tokens)
        context_embeddings = F.dropout(
            context_embeddings, p=self.dropout, training=self.training)
        question_embeddings = F.dropout(
            question_embeddings, p=self.dropout, training=self.training)

        if self.question_embed:
            context_hiddens = self.context_question_attention(
                context_embeddings, question_embeddings)
            context_embeddings = torch.cat(
                [context_embeddings, context_hiddens], dim=2)

        # Combine with engineered features
        if self.num_features > 0 and 'context_features' in kwargs:
            context_embeddings = torch.cat(
                [context_embeddings, kwargs['context_features']], dim=2)

        # Encode context words and question words with RNNs
        context_hiddens = self.context_rnn(context_embeddings, context_mask)
        question_hiddens = self.question_rnn(
            question_embeddings, question_mask)

        # Summarize hidden states of question words into a vector
        attention_scores = self.question_attention(
            question_hiddens, log_probs=False)
        question_hidden = (attention_scores.unsqueeze(dim=2)
                           * question_hiddens).sum(dim=1)

        # Predict answers with attentions
        start_scores = self.start_attention(
            question_hidden, context_hiddens, context_mask, log_probs=self.training)
        end_scores = self.end_attention(
            question_hidden, context_hiddens, context_mask, log_probs=self.training)
        return start_scores, end_scores

    @staticmethod
    def decode(start_scores, end_scores, topk=1, max_len=None):
        """Take argmax of constrained start_scores * end_scores."""
        pred_start, pred_end, pred_score = [], [], []
        max_len = max_len or start_scores.size(1)

        for i in range(start_scores.size(0)):
            # Outer product of scores to get a full matrix
            scores = torch.ger(start_scores[i], end_scores[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            max_scores, max_idx = scores.view(-1).topk(topk)
            pred_score.append(max_scores)
            pred_start.append(max_idx // scores.size(0))
            pred_end.append(max_idx % scores.size(0))
        return pred_start, pred_end, pred_score
