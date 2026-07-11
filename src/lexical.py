"""Word-level BM25 (sparse) retriever — hand-implemented (no rank_bm25).

Complements the char-n-gram encoder: BM25 rewards exact token overlap (street
names, house numbers), char-n-gram is robust to typos. Fusing both (RRF) is the
hybrid of the plan.
"""
import math
import re
from collections import defaultdict, Counter

_TOKEN = re.compile(r"[0-9a-zA-Zа-яёА-ЯЁ]+", re.UNICODE)


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, corpus):
        self.docs = [tokenize(t) for t in corpus]
        self.N = len(self.docs)
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / max(1, self.N)
        self.postings = defaultdict(list)      # term -> [(doc, tf)]
        self.df = defaultdict(int)
        for i, d in enumerate(self.docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
                self.df[term] += 1
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                    for t, df in self.df.items()}
        return self

    def search(self, query, k=10):
        scores = defaultdict(float)
        for term in dict.fromkeys(tokenize(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc] / self.avgdl)
                scores[doc] += idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores, key=lambda d: (-scores[d], d))[:k]
        return ranked
