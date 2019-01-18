

import torch
import ujson
import gzip
import string
import random
import sys
import pickle
import logging
import click

import numpy as np

from glob import glob
from tqdm import tqdm
from collections import Counter, defaultdict
from itertools import chain, islice

from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import rnn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from headline_parser import parse_headline


logging.basicConfig(
    format='%(asctime)s | %(levelname)s : %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('span-clf.log'),
    ]
)

logger = logging.getLogger('span-clf')


def read_json_gz_lines(root):
    """Read JSON corpus.

    Yields: dict
    """
    for path in glob('%s/*.gz' % root):
        with gzip.open(path) as fh:
            for line in fh:
                yield ujson.loads(line)


def group_by_sizes(L, sizes):
    """Given a flat list and a list of sizes that sum to the length of the
    list, group the list into sublists with corresponding sizes.

    Args:
        L (list)
        sizes (list<int>)

    Returns: list<list>
    """
    parts = []

    total = 0
    for s in sizes:
        parts.append(L[total:total+s])
        total += s

    return parts


DEVICE = (torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu'))


class Corpus:

    @classmethod
    def from_spark_lines(cls, root, skim=None):
        """Read JSON gz lines.
        """
        rows_iter = islice(read_json_gz_lines(root), skim)

        # Label -> [line1, line2, ...]
        groups = defaultdict(list)
        for row in tqdm(rows_iter):

            doc = parse_headline(row['title'])
            spans = [s._.clf_text for s in doc._.spans if s.text]

            # Only care about cases where we can remove a split.
            if len(spans) > 1:
                for span in spans:
                    groups[row['domain']].append(span)

        return cls(groups)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    def __init__(self, groups, test_frac=0.1):
        self.groups = groups
        self.test_frac = test_frac
        self.set_splits()

    def labels(self):
        return list(self.groups)

    def min_label_count(self):
        return min([len(v) for v in self.groups.values()])

    def set_splits(self):
        """Balance classes, fix train/val/test splits.
        """
        min_count = self.min_label_count()

        pairs = list(chain(*[
            [(line, label) for line in random.sample(lines, min_count)]
            for label, lines in self.groups.items()
        ]))

        test_size = round(len(pairs) * self.test_frac)
        train_size = len(pairs) - (test_size * 2)
        sizes = (train_size, test_size, test_size)

        self.train, self.val, self.test = random_split(pairs, sizes)


class CharEmbedding(nn.Embedding):

    def __init__(self, embed_dim=15):
        """Set vocab, map s->i.
        """
        self.vocab = (
            string.ascii_letters +
            string.digits +
            string.punctuation
        )

        # <UNK> -> 1
        self._ctoi = {s: i+1 for i, s in enumerate(self.vocab)}

        super().__init__(len(self.vocab)+1, embed_dim)

    @property
    def out_dim(self):
        return self.weight.shape[1]

    def ctoi(self, c):
        return self._ctoi.get(c, 0)

    def chars_to_idxs(self, chars):
        """Map characters to embedding indexes.
        """
        idxs = [self.ctoi(c) for c in chars]

        return torch.LongTensor(idxs).to(DEVICE)

    def forward(self, chars):
        """Batch-embed chars.

        Args:
            tokens (list<str>)
        """
        idxs = [self.ctoi(c) for c in chars]
        x = torch.LongTensor(idxs).to(DEVICE)

        return super().forward(x)


class SpanEncoder(nn.Module):

    def __init__(self, input_size, hidden_size=512, num_layers=1):
        """Initialize LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.out_dim = self.lstm.hidden_size * 2

    def forward(self, xs):
        """Sort, pack, encode, reorder.

        Args:
            xs (list<Tensor>): Variable-length embedding tensors.

        Returns:
            x (Tensor): F/B hidden tops.
        """
        sizes = [len(x) for x in xs]

        # Indexes to sort descending.
        sort_idxs = np.argsort(sizes)[::-1]

        # Indexes to restore original order.
        unsort_idxs = torch.from_numpy(np.argsort(sort_idxs)).to(DEVICE)

        # Sort by size descending.
        xs = [xs[i] for i in sort_idxs]

        # Pad + pack, LSTM.
        x = rnn.pack_sequence(xs)
        _, (hn, _) = self.lstm(x)

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[unsort_idxs]

        return x


class Classifier(nn.Module):

    def __init__(self, labels, hidden_dim=256):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.labels = labels
        self.ltoi = {label: i for i, label in enumerate(labels)}

        self.embed_chars = CharEmbedding()
        self.encode_spans = SpanEncoder(self.embed_chars.out_dim)

        self.dropout = nn.Dropout()

        self.predict = nn.Sequential(
            nn.Linear(self.encode_spans.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(labels)),
            nn.LogSoftmax(1),
        )

    def forward(self, spans):
        """Predict outlet.
        """
        sizes = [len(s) for s in spans]

        # Embed chars, regroup by line.
        x = self.embed_chars(list(chain(*spans)))
        xs = group_by_sizes(x, sizes)

        # Embed spans.
        x = self.encode_spans(xs)
        x = self.dropout(x)

        return self.predict(x)

    def collate_batch(self, batch):
        """Labels -> indexes.
        """
        lines, labels = list(zip(*batch))

        yt_idx = [self.ltoi[label] for label in labels]
        yt = torch.LongTensor(yt_idx).to(DEVICE)

        return lines, yt


def train_epoch(model, optimizer, loss_func, split):

    loader = DataLoader(
        split,
        collate_fn=model.collate_batch,
        batch_size=50,
    )

    losses = []
    for spans, yt in tqdm(loader):

        model.train()
        optimizer.zero_grad()

        yp = model(spans)

        loss = loss_func(yp, yt)
        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return losses


def predict(model, split):

    model.eval()

    loader = DataLoader(
        split,
        collate_fn=model.collate_batch,
        batch_size=50,
    )

    yt, yp = [], []
    for lines, yti in loader:
        yp += model(lines).tolist()
        yt += yti.tolist()

    yt = torch.LongTensor(yt)
    yp = torch.FloatTensor(yp)

    return yt, yp


def evaluate(model, loss_func, split):
    yt, yp = predict(model, split)
    return loss_func(yp, yt)



@click.group()
def cli():
    pass


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--skim', type=int, default=None)
def build_corpus(src, dst, skim):
    corpus = Corpus.from_spark_lines(src, skim)
    corpus.save(dst)


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--max_epochs', type=int, default=100)
@click.option('--es_wait', type=int, default=5)
def train(src, dst, max_epochs, es_wait):
    """Train, dump model.
    """
    corpus = Corpus.load(src)

    model = Classifier(corpus.labels())
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    losses = []
    for _ in range(max_epochs):

        train_epoch(model, optimizer, loss_func, corpus.train)

        loss = evaluate(model, loss_func, corpus.val)
        losses.append(loss)

        logger.info(loss.item())

        if len(losses) > es_wait and losses[-1] > losses[-es_wait]:
            break

    torch.save(model, dst)


if __name__ == '__main__':
    cli()
