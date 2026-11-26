"""Text preprocessing for Khmer reviews (written by notebook 01_data_preparation.ipynb)."""
import logging
import re
import string
import unicodedata

from khmernltk import word_tokenize

logging.getLogger("khmer-nltk").setLevel(logging.WARNING)   # hide the "Loaded model from <path>" message

ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
URL = re.compile(r"https?://\S+|www\.\S+")
REPEATS = re.compile(r"(.{1,6}?)\1{3,}")              # a piece of 1-6 characters repeated 4+ times
KHMER_PUNCTUATION = "\u17d4\u17d5\u17d6\u17d8\u17d9\u17da"   # ។ ៕ ៖ ៘ ៙ ៚  (ៗ is kept: it repeats a word)
PUNCTUATION = re.compile("[" + re.escape(string.punctuation + "“”‘’«»…–—•·") + KHMER_PUNCTUATION + "]")
NUMBERS = re.compile("[0-9\u17e0-\u17e9]+")          # 0-9 and Khmer digits ០-៩


def preprocess_text(text):
    # 1. Unicode normalization: the same letter is always stored the same way
    text = unicodedata.normalize("NFC", str(text))

    # 2. Remove zero-width characters and URLs
    text = ZERO_WIDTH.sub("", text)
    text = URL.sub(" ", text)

    # 3. Collapse 4+ repetitions to 3 ("sooooo" -> "sooo")
    text = REPEATS.sub(r"\1\1\1", text)

    # 4. Lowercase English words (code-switching)
    text = text.lower()

    # 5. Remove punctuation and numbers, but keep emojis
    text = PUNCTUATION.sub(" ", text)
    text = NUMBERS.sub(" ", text)

    # 6. Cut Khmer into words with khmer-nltk, and drop the empty pieces (spaces)
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens if token.strip()]

    # No stopword removal: negation words (មិន, ទេ, គ្មាន, កុំ, ពុំ, not, no) are kept
    return tokens   # a list of words
