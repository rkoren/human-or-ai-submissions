"""Model factories shared across experiments (ported from notebooks/02_init_model.ipynb).

Every factory returns an unfitted sklearn pipeline that takes a Series of raw texts and
exposes predict_proba, so the same object works for Q1-Q5 scripts.
"""
import re

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer

try:
    from nltk.corpus import stopwords
    FUNCTION_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    FUNCTION_WORDS = set(stopwords.words("english"))

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|\n|[^\w\s]")


def mask_content(text):
    """Hide topic words: keep function words, '*' for content words, '#' for numbers, '^' for newlines."""
    masked = []
    for tok in TOKEN.findall(text):
        if tok == "\n":
            masked.append("^")
        elif tok[0].isdigit():
            masked.append("#")
        elif tok[0].isalpha():
            masked.append(tok.lower() if tok.lower() in FUNCTION_WORDS else "*")
        else:
            masked.append(tok)
    return " ".join(masked)


def mask_texts(texts):
    return texts.map(mask_content)


def flatten(texts):
    return texts.str.replace(r"\s+", " ", regex=True).str.strip()


def make_tfidf_lr(C=4.0):
    """Word 1-2gram + char 3-5gram TF-IDF, logistic regression (baseline from notebook 02)."""
    return make_pipeline(
        make_union(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=200_000),
        ),
        LogisticRegression(C=C, max_iter=3000),
    )


def make_masked_lr(C=4.0, flat=False):
    """Masked 1-3gram TF-IDF + logistic regression. flat=True removes all linebreak information."""
    steps = [FunctionTransformer(flatten)] if flat else []
    steps += [
        FunctionTransformer(mask_texts),
        TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 3), min_df=3,
                        sublinear_tf=True, max_features=200_000),
        LogisticRegression(C=C, max_iter=3000),
    ]
    return make_pipeline(*steps)


MODELS = {
    "tfidf_lr": make_tfidf_lr,
    "masked_lr": make_masked_lr,
    "masked_flat_lr": lambda: make_masked_lr(flat=True),
}
