"""Herbruikbare tokenizers voor de recommender.

Dezelfde functies als in 03 bow_tfidf.ipynb, maar dan op één plek. Gebruik:

    from functions import tokenize_nostop
    TfidfVectorizer(tokenizer=tokenize_nostop, token_pattern=None, lowercase=False)

Alle tokenizers geven een lijst woorden terug. lowercase=False in de vectorizer is bewust:
clean() doet het kleine-letters-werk al, en token_pattern=None voorkomt een waarschuwing
dat het standaardpatroon genegeerd wordt.
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# --- Stopwoorden -------------------------------------------------------------------------
# NLTK-lijst wordt één keer gedownload, daarna uit de lokale cache gelezen.
nltk.download("stopwords", quiet=True)
stop_nl: set[str] = set(stopwords.words("dutch"))

# Eigen aanvulling: woorden die in deze catalogus overal in staan en dus niets onderscheiden.
# Dit is een keuze; noteer hem in het keuzelog.
CUSTOM_STOPWORDS: set[str] = {"boek", "verhaal", "jongen", "meisje", "jaar"}
all_stop_words: set[str] = stop_nl | CUSTOM_STOPWORDS

# --- Stemming ----------------------------------------------------------------------------
stemmer = SnowballStemmer("dutch")
stem = stemmer.stem


# --- Tokenizers --------------------------------------------------------------------------
def clean(text: str) -> str:
    """Kleine letters; verder niets. Vangt ook NaN op (die wordt een lege string)."""
    return str(text).lower() if text == text else ""   # NaN != NaN


def tokenize(text: str) -> list[str]:
    """Basis: losse woorden (letters, cijfers, underscore), alles in kleine letters."""
    return re.findall(r"\w+", clean(text))


def tokenize_stop(text: str) -> list[str]:
    """Basis minus de NLTK-stopwoorden."""
    return [w for w in tokenize(text) if w not in stop_nl]


def tokenize_nostop(text: str) -> list[str]:
    """Basis minus NLTK-stopwoorden én de eigen aanvulling (CUSTOM_STOPWORDS)."""
    return [w for w in tokenize(text) if w not in all_stop_words]


def tokenize_stemmed(text: str) -> list[str]:
    """Basis, elk woord teruggebracht tot zijn stam ("draken" -> "drak")."""
    return [stem(w) for w in tokenize(text)]


def tokenize_stemmed_nostop(text: str) -> list[str]:
    """Stopwoorden eruit, daarna elk woord naar zijn stam."""
    return [stem(w) for w in tokenize_nostop(text)]

# --- Een paar titels naast elkaar: tekstblok boven, categorieblokken onder --------------------
def show_vectors(title_list, X, columns, books, n_words=5, text_col="Korte omschrijving"):
    """Toon voor een paar titels hoe hun vector eruitziet.

    Boven: per titel de n zwaarste woorden uit het tekstblok. Onder: alle categoriekolommen
    (genre, trefwoorden, ...) die bij minstens één van deze titels gevuld zijn. Nullen worden
    leeg gelaten, zodat de gevulde cellen eruit springen. Een categoriecel bevat precies het
    gewicht van dat blok; een titel zonder genre heeft daar niets staan.

    columns: de kolomnamen zoals build_vectors_full ze teruggeeft, in de vorm "<kolom>=<waarde>".
    """
    titles = books["Titel"].tolist()
    rows = [titles.index(t) for t in title_list]
    dense = pd.DataFrame(X[rows].toarray(), index=title_list, columns=columns)

    prefix = text_col + "="
    word_cols_all = [c for c in columns if c.startswith(prefix)]
    cat_cols = [c for c in columns if not c.startswith(prefix)]

    word_cols = []
    for t in title_list:                                   # top-n woorden per titel, zonder dubbelen
        top = dense.loc[t, word_cols_all].nlargest(n_words).index
        word_cols += [w for w in top if w not in word_cols]
    cat_used = [c for c in cat_cols if (dense[c] != 0).any()]   # alleen categorieën die hier voorkomen

    table = dense[word_cols + cat_used].round(2).T         # features als rijen, titels als kolommen
    table.index = [c.replace(prefix, "") for c in table.index]
    return table.replace(0, "")
