"""Classical AI-generated-text (LLM) detector.

Eight independent forensic signals, no learned models. Each returns a score in
[0, 1] where higher = more AI-like (machine-generated). The signals key on the
*statistical regularities* large language models leave in their output relative
to human prose, all measurable from the raw string with no model and no
network. This mirrors ``detector.py`` (image track): a fixed-weight linear
combiner over interpretable signals as the training-free baseline.

Signals:
  burstiness         - low sentence-length variance => AI (humans are bursty)
  ttr                - low type-token ratio (lexical diversity) => AI
  ngram_repetition   - high repeated bi/tri-gram rate => AI
  function_word_reg  - over-regular function-word ratios => AI
  punct_diversity    - narrow punctuation variety => AI
  rare_word_rarity   - under-use of rare/long words => AI
  transition_density - boilerplate connective density => AI
  token_entropy      - low Shannon entropy of token distribution => AI (mild)

Every signal is robust to short/empty/degenerate input: it returns the neutral
value 0.5 when there is not enough text to measure the statistic.
"""

from __future__ import annotations

import math
import re
from collections import Counter


# --------------------------------------------------------------------------- #
# tokenisation helpers
# --------------------------------------------------------------------------- #

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+(?:\s+|$)")

# Closed-class function words: their *relative proportions* are remarkably
# stable in AI text, which over-regularises grammatical scaffolding.
_FUNCTION_WORDS = frozenset(
    """a an the and or but nor so yet for of to in on at by with from into onto
    over under above below up down out off as is are was were be been being am
    do does did have has had this that these those it its it's he she they we
    you i him her them us me my your his their our not no can will would should
    could may might must shall about than then there here which who whom whose
    what when where why how if because while although though since until""".split()
)

# Boilerplate connectives / discourse markers that LLMs favour heavily.
_TRANSITION_PHRASES = (
    "furthermore", "moreover", "in conclusion", "it is important to note",
    "additionally", "overall", "however", "therefore", "consequently",
    "in summary", "on the other hand", "as a result", "first and foremost",
    "in addition", "notably", "importantly", "ultimately", "in essence",
    "it is worth noting", "to summarize", "in other words",
)

# A small built-in list of the most common English words. A word NOT in this
# set is treated as "rare". No download — this is a fixed in-code proxy for a
# frequency table, matching the image track's self-contained-corpus philosophy.
_COMMON_WORDS = frozenset(
    """the be to of and a in that have i it for not on with he as you do at
    this but his by from they we say her she or an will my one all would there
    their what so up out if about who get which go me when make can like time
    no just him know take people into year your good some could them see other
    than then now look only come its over think also back after use two how our
    work first well way even new want because any these give day most us is are
    was were been being am has had did does said made many such very through
    where much before great little world still own under last right too should
    while between those both each may might must own same few here every left
    around three things place again away old men number off found long down
    while life away both got while same"""
    .split()
)


def _words(text: str) -> list[str]:
    """Lower-cased alphabetic word tokens."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    """Split into sentences on terminal punctuation; drop empties."""
    parts = [s.strip() for s in _SENT_SPLIT_RE.split(text)]
    return [s for s in parts if s]


def _sentence_lengths(text: str) -> list[int]:
    """Word counts per sentence."""
    return [len(_WORD_RE.findall(s)) for s in _sentences(text) if _WORD_RE.findall(s)]


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# --------------------------------------------------------------------------- #
# signals
# --------------------------------------------------------------------------- #


def burstiness(text: str) -> float:
    """Sentence-length burstiness.

    Human writing is *bursty*: it interleaves long and short sentences, giving a
    high coefficient of variation (CV = std / mean) of sentence length. LLMs
    tend toward a uniform, metronomic sentence length, so a LOW CV indicates AI.
    Score = 1 - clip(CV / 0.6): CV >= 0.6 (very human) -> ~0, CV -> 0 (uniform)
    -> 1.
    """
    lengths = _sentence_lengths(text)
    if len(lengths) < 2:
        return 0.5
    mean = sum(lengths) / len(lengths)
    if mean <= 0:
        return 0.5
    var = sum((x - mean) ** 2 for x in lengths) / len(lengths)
    cv = math.sqrt(var) / mean
    return _clip01(1.0 - cv / 0.6)


def ttr(text: str) -> float:
    """Type-token ratio (lexical diversity).

    Distinct-words / total-words. Humans draw on a wider vocabulary per unit of
    text; LLMs (especially under greedy/low-temperature decoding) recycle words,
    lowering the TTR. To remove the well-known length bias of raw TTR, the ratio
    is measured on a fixed 100-token window. Low TTR => AI.
    Score = 1 - clip((TTR - 0.4) / 0.4): TTR >= 0.8 -> 0, TTR <= 0.4 -> 1.
    """
    w = _words(text)
    if len(w) < 20:
        return 0.5
    window = w[:100]
    ratio = len(set(window)) / len(window)
    return _clip01(1.0 - (ratio - 0.4) / 0.4)


def ngram_repetition(text: str) -> float:
    """Repeated n-gram rate (bigrams + trigrams).

    LLMs degenerate into repeating phrases and structural templates far more
    than humans do (Holtzman et al. 2020, "neural text degeneration"). We
    measure the fraction of bi- and tri-grams that are NOT unique (i.e. repeat
    at least once). A high repeated-gram fraction => AI.
    Score = clip(repeat_fraction / 0.25): 25%+ repeated grams -> 1.
    """
    w = _words(text)
    if len(w) < 6:
        return 0.5
    fracs = []
    for n in (2, 3):
        grams = [tuple(w[i : i + n]) for i in range(len(w) - n + 1)]
        if not grams:
            continue
        counts = Counter(grams)
        repeated = sum(c for c in counts.values() if c > 1)
        fracs.append(repeated / len(grams))
    if not fracs:
        return 0.5
    return _clip01((sum(fracs) / len(fracs)) / 0.25)


def function_word_regularity(text: str) -> float:
    """Function-word over-regularisation.

    Function words (the/of/and/to ...) form the grammatical skeleton of text.
    Their *overall density* among all tokens is one of the most stable stylistic
    fingerprints. AI text tends to land on a narrow, "textbook" function-word
    density (~0.45-0.55 of tokens), whereas human text spreads more widely.
    We score by closeness to that AI-typical density: |density - 0.50| small =>
    AI. Score = 1 - clip(|density - 0.50| / 0.25).
    """
    w = _words(text)
    if len(w) < 15:
        return 0.5
    fw = sum(1 for x in w if x in _FUNCTION_WORDS)
    density = fw / len(w)
    return _clip01(1.0 - abs(density - 0.50) / 0.25)


def punctuation_diversity(text: str) -> float:
    """Punctuation-variety narrowness.

    Human writers reach for a broad punctuation palette — semicolons, dashes,
    parentheses, colons, ellipses — while LLM prose leans on a narrow set
    (mostly periods and commas). Low distinct-punctuation variety => AI.
    Variety is normalised by text length (so long human text isn't penalised
    for using each mark once). Score = 1 - clip(distinct_marks / 6).
    """
    marks = re.findall(r"[.,;:!?\-—()\"'…]", text)
    if len(text.strip()) < 40 or not marks:
        return 0.5
    distinct = len(set(marks))
    return _clip01(1.0 - distinct / 6.0)


def rare_word_rarity(text: str) -> float:
    """Rare-word under-use.

    Human prose sprinkles in rare and long words; LLMs gravitate to high-
    frequency, "safe" vocabulary, depressing the rare-word rate. Rarity is
    proxied two ways (no external frequency table): a word is "rare" if it is
    absent from the built-in common-word list, OR is long (>= 9 letters). A LOW
    rare-word fraction => AI. Score = 1 - clip(rare_fraction / 0.45).
    """
    w = _words(text)
    if len(w) < 15:
        return 0.5
    rare = sum(1 for x in w if (x not in _COMMON_WORDS) or len(x) >= 9)
    frac = rare / len(w)
    return _clip01(1.0 - frac / 0.45)


def transition_phrase_density(text: str) -> float:
    """Boilerplate-connective density.

    LLMs over-produce discourse scaffolding ("furthermore", "moreover", "in
    conclusion", "it is important to note", ...). We count phrase occurrences
    per 100 words. Higher density => AI.
    Score = clip(per_100_words / 2.0): ~2 boilerplate phrases / 100 words -> 1.
    """
    w = _words(text)
    if len(w) < 15:
        return 0.5
    low = text.lower()
    hits = sum(low.count(p) for p in _TRANSITION_PHRASES)
    per100 = hits / (len(w) / 100.0)
    return _clip01(per100 / 2.0)


def token_entropy(text: str) -> float:
    """Token-distribution Shannon entropy (mild signal).

    The normalised Shannon entropy of the word-frequency distribution measures
    how evenly probability mass is spread across the vocabulary. AI text, with
    its recycled vocabulary, concentrates mass and so has slightly LOWER
    normalised entropy than diverse human text. Score = 1 - clip((H - 0.6) /
    0.4): H_norm >= 1.0 -> 0, H_norm <= 0.6 -> 1.
    """
    w = _words(text)
    if len(w) < 20:
        return 0.5
    counts = Counter(w)
    n = len(w)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    max_h = math.log2(len(counts)) if len(counts) > 1 else 1.0
    h_norm = h / max_h if max_h > 0 else 1.0
    return _clip01(1.0 - (h_norm - 0.6) / 0.4)


# --------------------------------------------------------------------------- #
# combiner
# --------------------------------------------------------------------------- #


SIGNAL_NAMES = (
    "burstiness",
    "ttr",
    "ngram_repetition",
    "function_word_reg",
    "punct_diversity",
    "rare_word_rarity",
    "transition_density",
    "token_entropy",
)

DEFAULT_WEIGHTS = (0.18, 0.15, 0.15, 0.12, 0.08, 0.14, 0.12, 0.06)

_SIGNAL_FNS = (
    burstiness,
    ttr,
    ngram_repetition,
    function_word_regularity,
    punctuation_diversity,
    rare_word_rarity,
    transition_phrase_density,
    token_entropy,
)


def all_scores(text: str) -> dict[str, float]:
    """Compute every forensic signal for one text, keyed by SIGNAL_NAMES."""
    return {name: fn(text) for name, fn in zip(SIGNAL_NAMES, _SIGNAL_FNS)}


def detect(text: str, weights: tuple[float, ...] = DEFAULT_WEIGHTS) -> dict:
    """Fixed-weight linear combiner over the signal bank.

    Returns each signal plus ``combined`` (weighted mean in [0, 1]) and a
    ``verdict`` of ``"ai"`` (combined > 0.5) or ``"human"``.
    """
    s = all_scores(text)
    combined = float(sum(w * s[name] for w, name in zip(weights, SIGNAL_NAMES)))
    return {**s, "combined": combined, "verdict": "ai" if combined > 0.5 else "human"}


if __name__ == "__main__":
    _human = (
        "This is a short human-written sentence with some variety. It rambles a "
        "bit, then stops."
    )
    _ai = (
        "Furthermore, it is important to note that the system is robust. "
        "Moreover, the system is robust and reliable. Additionally, the system "
        "is robust and reliable. In conclusion, the system is robust and "
        "reliable overall."
    )
    print("human-ish:", detect(_human))
    print("ai-ish:   ", detect(_ai))
    print("empty:    ", detect(""))
