"""Self-supervised pseudo-label generation for label-free AI-text detection.

The text detector is trained *without any hand-labelled AI text*. We take a
corpus of ordinary human-written prose (treated as the "real"/human manifold)
and synthesise two views of each source:

  * a **pseudo-human** view - a benign, label-preserving edit only (whitespace
    and case normalisation plus a light punctuation tweak), the kind of edit a
    real document legitimately undergoes.
  * a **pseudo-AI** view   - the same text with one or two *LLM-style statistical
    artifacts* injected. These transforms emulate, in a controlled and fully
    self-supervised way, the traces large language models leave behind.

This mirrors ``selfsup.py`` (image track): supervision is *constructed*, not
annotated. A classifier that learns to separate the two views keys on the
*statistical artifact* of machine generation, not on any specific model.

The built-in HUMAN seed corpus is a small set of varied paragraphs written for
this file (literary, conversational, technical, news-like). Exactly like the
image pipeline's tiny stand-in corpus, this is an honest stand-in for the real
human manifold, not a claim of broad coverage — AUC on so few independent
scenes is optimistic (documented in evaluate_text.py).

Pseudo-AI transforms (each randomised per sample):
  flatten_burstiness    - rewrite toward uniform sentence length by splitting
                          long sentences and merging short ones (kills the
                          human long/short cadence LLMs lack).
  inject_repetition     - repeat phrases / n-grams (neural-degeneration trace).
  inject_boilerplate    - prepend LLM discourse connectives to sentences.
  lexical_smoothing     - replace rare/long words with common synonyms (the
                          vocabulary-narrowing of low-temperature decoding).
  regularize_punctuation- collapse the punctuation palette to periods/commas.

All transforms are pure-Python and deterministic given a seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- #
# built-in human seed corpus (the unlabelled "human" manifold stand-in)
# --------------------------------------------------------------------------- #

HUMAN_CORPUS: list[str] = [
    # literary
    "The harbour smelled of salt and diesel that morning. Gulls wheeled. An old "
    "man mended his nets with hands cracked like dry riverbeds, humming a tune "
    "nobody had sung in forty years, while the tide crept in around the pilings "
    "and the light went from grey to a thin, reluctant gold.",
    # conversational
    "So I told him, look, I'm not doing that again. Last time we both ended up "
    "soaked and the dog ran off. He laughed. He always laughs when I'm serious. "
    "Anyway, we got the dog back, eventually, hiding under the neighbour's "
    "porch like the little coward he is. Good boy though.",
    # technical
    "A hash table trades memory for speed. Insertions and lookups run in "
    "amortised constant time, assuming a decent hash and a load factor kept "
    "below roughly seventy percent. Collisions are inevitable; chaining and "
    "open addressing handle them differently, and each fails in its own "
    "characteristic, irritating way under adversarial keys.",
    # news-like
    "City council voted late Tuesday to delay the bridge repair, citing budget "
    "shortfalls. Residents who packed the chamber were furious. \"We've waited "
    "three years,\" one said. Officials promised a revised timeline by spring, "
    "though few in the room seemed willing to believe it after the last two.",
    # reflective essay
    "Memory is unreliable, and yet we build whole lives on it. I remember the "
    "kitchen of my grandmother's house more vividly than my own from last year. "
    "Was the wallpaper really that yellow? Probably not. But the mind keeps what "
    "it wants, embellishes the rest, and calls the result the truth.",
    # how-to
    "Start with cold butter, not room temperature. Cut it into the flour until "
    "the mixture looks like coarse breadcrumbs. Add the ice water a spoonful at "
    "a time. Stop the moment it holds together. Overwork it and you get "
    "cardboard; treat it gently and the pastry shatters into flakes.",
    # historical
    "By the autumn of 1929 the warnings had been ignored for months. Brokers "
    "kept lending. Clerks kept buying on margin. When the market finally broke, "
    "it broke all at once, and the fortunes built on borrowed optimism "
    "evaporated in an afternoon, taking ordinary households down with them.",
    # nature writing
    "The forest changes its voice at dusk. Birdsong thins; something larger "
    "moves through the undergrowth without hurry. A fox, perhaps, or only the "
    "wind testing the brambles. You learn, after enough nights out here, to "
    "tell the difference, and to stop being frightened by the wrong one.",
    # opinion column
    "We keep mistaking motion for progress. The new app launches, the metrics "
    "spike, everyone claps. Six months later nobody uses it and we have learned "
    "nothing, because we never asked the only question that mattered: did this "
    "make anyone's day even slightly better? Usually the honest answer is no.",
    # science explainer
    "Light bends when it passes from air into water because it slows down, and "
    "the part of the wavefront that enters first is held back while the rest "
    "races ahead. That tilt is refraction. It is why a straw looks broken in a "
    "glass, and why your legs seem shorter in the shallow end of a pool.",
    # travel
    "The train climbed for hours through valleys still half-asleep under fog. "
    "Villages slid past, each with its single church and its scatter of red "
    "roofs. An old woman across the aisle offered me bread and said nothing, "
    "and somehow that small kindness mattered more than the whole grand view.",
    # personal anecdote
    "My first job paid almost nothing and taught me almost everything. I "
    "filed, I fetched coffee, I watched. The man who ran the place never "
    "explained himself, but I copied what he did until I understood why he did "
    "it. Years later I caught myself doing the same to a younger colleague.",
    # sports
    "With two minutes left they were down by four and the crowd had mostly "
    "given up. Then the kid nobody had heard of stole the ball, drove the "
    "length of the court, and sank a shot that had no business going in. The "
    "place came apart. People who'd left came running back inside.",
    # philosophy
    "To say a thing is good is not merely to report a fact about it. It is to "
    "recommend it, to take a stance, to invite agreement. This is why moral "
    "arguments rarely end the way mathematical ones do: we are not just "
    "describing the world, we are quarrelling about how to live in it.",
    # food review
    "The broth arrives almost black, glossy, fragrant with star anise and "
    "something faintly bitter underneath. The noodles have bite. The beef "
    "falls apart at a glance. It is not a refined dish and it does not pretend "
    "to be; it is the kind of bowl you finish and then quietly order again.",
    # letter
    "Dear Sam, the garden survived the winter, barely. Two of the roses are "
    "gone but the apple tree is heavy with blossom and the bees have found it "
    "already. I keep your old chair on the porch. Some evenings I sit in it and "
    "talk to you as if you might answer. Foolish, I know. Write soon.",
    # mystery
    "The door was locked from the inside. The window had not been opened in "
    "years, its paint sealing it shut. And yet the room was empty, the bed cold, "
    "the man who had screamed for help simply gone. The inspector lit a "
    "cigarette, said nothing for a long while, and began, slowly, to smile.",
    # economics
    "Prices are signals, not commands. When coffee gets dearer, nobody orders "
    "anyone to drink less; people simply notice, grumble, and adjust. Multiply "
    "that quiet adjustment across millions of strangers who never meet, and you "
    "get a kind of order that no central planner could ever have designed.",
    # memoir
    "We were poor but I did not know it, which is its own kind of wealth. The "
    "house was loud and small and full. There was always someone arguing, "
    "someone cooking, someone crying over a thing that would be forgotten by "
    "supper. I would give a great deal to stand in that noise once more.",
    # tech opinion
    "Every interface is a tiny argument about how you should think. The blank "
    "page insists you have something to say. The endless feed insists you do "
    "not, that you should merely scroll and consume. We shape our tools, the "
    "old line goes, and thereafter our tools shape us, often badly.",
]


# --------------------------------------------------------------------------- #
# corpus -> sources (split into sentence windows for leakage-free grouping)
# --------------------------------------------------------------------------- #

_SENT_RE = re.compile(r"[^.!?]*[.!?]+", re.S)


def _split_sentences(text: str) -> list[str]:
    """Sentence segmentation that keeps terminal punctuation attached."""
    sents = [m.group(0).strip() for m in _SENT_RE.finditer(text)]
    sents = [s for s in sents if s]
    if not sents:                       # no terminal punctuation -> whole thing
        s = text.strip()
        return [s] if s else []
    return sents


def build_sources(
    corpus: list[str] | None = None,
    window: int = 2,
) -> list[tuple[str, str]]:
    """Expand the corpus into named *sources* of ``window`` sentences each.

    Splitting paragraphs into non-overlapping sentence windows multiplies the
    number of independent "scenes" available for self-supervision and for
    leakage-free grouped evaluation — directly analogous to patch tiling of
    images in ``selfsup.load_sources``. Each source gets a stable id
    ``"docDD#wWW"``. A trailing remainder shorter than ``window`` is kept as its
    own source so no text is discarded.
    """
    corpus = HUMAN_CORPUS if corpus is None else corpus
    sources: list[tuple[str, str]] = []
    for di, para in enumerate(corpus):
        sents = _split_sentences(para)
        if not sents:
            continue
        for wi, start in enumerate(range(0, len(sents), window)):
            chunk = " ".join(sents[start : start + window]).strip()
            if chunk:
                sources.append((f"doc{di:02d}#w{wi:02d}", chunk))
    return sources


# --------------------------------------------------------------------------- #
# benign (pseudo-human) edit — light, label-preserving
# --------------------------------------------------------------------------- #


def benign_edit(text: str, rng: np.random.Generator) -> str:
    """A label-preserving edit a real human document might undergo.

    Whitespace/case normalisation plus, optionally, one light punctuation tweak
    (a comma swapped for a semicolon, or an added space after punctuation). It
    never changes word choice, sentence cadence, or vocabulary, so the human
    label is preserved. The SAME benign edit family is applied to BOTH views so
    the classifier cannot cheat on the benign distribution alone.
    """
    out = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    out = re.sub(r"\s+([.,;:!?])", r"\1", out)          # no space before punct
    out = re.sub(r"([.,;:!?])(?=[^\s])", r"\1 ", out)   # one space after punct
    if rng.random() < 0.4:                              # light punctuation tweak
        commas = [m.start() for m in re.finditer(r",", out)]
        if commas:
            i = int(rng.choice(commas))
            out = out[:i] + ";" + out[i + 1 :]
    if rng.random() < 0.3:                              # capitalise first letter
        out = out[:1].upper() + out[1:] if out else out
    return out


# --------------------------------------------------------------------------- #
# pseudo-AI artifact-injection transforms
# --------------------------------------------------------------------------- #

# A small built-in rare->common synonym map for lexical smoothing (the
# vocabulary-narrowing trace of conservative LLM decoding). No external data.
_SYNONYM_MAP = {
    "reluctant": "slow", "irritating": "annoying", "characteristic": "typical",
    "adversarial": "hard", "embellishes": "adds to", "evaporated": "vanished",
    "undergrowth": "bushes", "frightened": "scared", "refraction": "bending",
    "wavefront": "wave", "scatter": "group", "quarrelling": "arguing",
    "fragrant": "smelling", "blossom": "flowers", "inspector": "officer",
    "central": "main", "thereafter": "later", "amortised": "average",
    "characteristic,": "typical,", "reluctant,": "slow,", "vividly": "clearly",
    "embellishes,": "adds,", "optimism": "hope", "ordinary": "normal",
    "kindness": "good deed", "refined": "fancy", "faintly": "slightly",
}

_BOILERPLATE = (
    "Furthermore", "Moreover", "Additionally", "In addition", "Importantly",
    "Notably", "It is important to note that", "Overall", "Consequently",
)


def flatten_burstiness(text: str, rng: np.random.Generator) -> str:
    """Rewrite toward uniform sentence length (kills human burstiness).

    Long sentences are split at a comma into two; consecutive short sentences
    are merged with a comma. The result has a flat, metronomic cadence — the
    low sentence-length variance that distinguishes LLM prose.
    """
    sents = _split_sentences(text)
    if not sents:
        return text
    out: list[str] = []
    for s in sents:
        words = s.split()
        if len(words) > 14 and "," in s:                # split long sentences
            commas = [i for i, ch in enumerate(s) if ch == ","]
            cut = commas[len(commas) // 2]
            left, right = s[:cut].strip(), s[cut + 1 :].strip()
            if left and right:
                left = left.rstrip(".!?") + "."
                right = right[:1].upper() + right[1:]
                if not right.endswith((".", "!", "?")):
                    right += "."
                out.extend([left, right])
                continue
        out.append(s)
    # merge adjacent short sentences
    merged: list[str] = []
    i = 0
    while i < len(out):
        cur = out[i]
        if (i + 1 < len(out) and len(cur.split()) < 6
                and len(out[i + 1].split()) < 6):
            nxt = out[i + 1]
            cur = cur.rstrip(".!?") + ", " + (nxt[:1].lower() + nxt[1:])
            i += 2
        else:
            i += 1
        merged.append(cur)
    return " ".join(merged)


def inject_repetition(text: str, rng: np.random.Generator) -> str:
    """Repeat an n-gram / phrase (neural text-degeneration trace).

    LLMs degenerate into repeating phrases far more than humans. We pick a short
    span and re-insert it once or twice, or duplicate a whole clause, raising the
    repeated-n-gram rate the detector keys on.
    """
    words = text.split()
    if len(words) < 6:
        return text + " " + text
    n = int(rng.integers(3, 6))
    start = int(rng.integers(0, max(1, len(words) - n)))
    phrase = " ".join(words[start : start + n])
    reps = int(rng.integers(1, 3))
    insert_at = int(rng.integers(0, len(words)))
    chunk = (" " + phrase) * reps
    new = " ".join(words[:insert_at]) + chunk + " " + " ".join(words[insert_at:])
    return re.sub(r"\s+", " ", new).strip()


def inject_boilerplate(text: str, rng: np.random.Generator) -> str:
    """Prepend LLM discourse connectives to sentences.

    Inserts boilerplate transition phrases ("Furthermore", "Moreover", "It is
    important to note that", ...) at the head of one or more sentences, raising
    the transition-phrase density typical of machine-generated prose.
    """
    sents = _split_sentences(text)
    if not sents:
        return text
    k = int(rng.integers(1, min(3, len(sents)) + 1))
    idxs = rng.choice(len(sents), size=k, replace=False)
    for i in idxs:
        marker = str(rng.choice(_BOILERPLATE))
        body = sents[i]
        if marker.endswith("that"):
            sents[i] = f"{marker} {body[:1].lower() + body[1:]}"
        else:
            sents[i] = f"{marker}, {body[:1].lower() + body[1:]}"
    return " ".join(sents)


def lexical_smoothing(text: str, rng: np.random.Generator) -> str:
    """Replace rare/long words with common synonyms (vocabulary narrowing).

    Maps a built-in set of rare or long words to plainer equivalents, emulating
    the way conservative (low-temperature) LLM decoding under-uses rare
    vocabulary. Lowers the rare-word and entropy signals toward AI.
    """
    def repl(m: re.Match) -> str:
        w = m.group(0)
        low = w.lower()
        if low in _SYNONYM_MAP and rng.random() < 0.9:
            sub = _SYNONYM_MAP[low]
            return sub[:1].upper() + sub[1:] if w[:1].isupper() else sub
        return w

    return re.sub(r"[A-Za-z]+", repl, text)


def regularize_punctuation(text: str, rng: np.random.Generator) -> str:
    """Collapse the punctuation palette to periods and commas.

    Converts semicolons/colons/dashes/parentheses/ellipses to commas or
    periods, narrowing punctuation variety the way much LLM prose does.
    """
    out = text
    out = out.replace("—", ", ").replace("–", ", ").replace("...", ".")
    out = out.replace("…", ".").replace(";", ",").replace(":", ",")
    out = out.replace("(", "").replace(")", "")
    out = re.sub(r",\s*,", ",", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


_AI_TRANSFORMS = {
    "flatten_burstiness": flatten_burstiness,
    "inject_repetition": inject_repetition,
    "inject_boilerplate": inject_boilerplate,
    "lexical_smoothing": lexical_smoothing,
    "regularize_punctuation": regularize_punctuation,
}


def make_pseudo_ai(
    text: str, rng: np.random.Generator, n_ops: tuple[int, int] = (1, 2)
) -> tuple[str, list[str]]:
    """Apply a random 1-2 LLM-style transforms in sequence."""
    k = int(rng.integers(n_ops[0], n_ops[1] + 1))
    names = [str(n) for n in rng.choice(list(_AI_TRANSFORMS), size=k, replace=False)]
    out = text
    for name in names:
        out = _AI_TRANSFORMS[name](out, rng)
    return out, names


# --------------------------------------------------------------------------- #
# sample generation
# --------------------------------------------------------------------------- #


@dataclass
class PseudoTextSample:
    text: str
    label: int          # 0 = pseudo-human, 1 = pseudo-AI
    source: str         # originating source id (doc#window)
    transforms: list[str] = field(default_factory=list)


def generate(
    per_source: int = 4,
    seed: int = 0,
    corpus: list[str] | None = None,
    window: int = 2,
) -> list[PseudoTextSample]:
    """Build a balanced self-supervised text dataset from the human corpus.

    For each source we emit ``per_source`` pseudo-human and ``per_source``
    pseudo-AI samples, giving a perfectly balanced set with no human labels.
    The pseudo-AI view also receives a benign edit (same family as the
    pseudo-human view) so the classifier cannot cheat on the benign
    distribution alone. Mirrors ``selfsup.generate``.
    """
    rng = np.random.default_rng(seed)
    sources = build_sources(corpus, window=window)
    samples: list[PseudoTextSample] = []
    for name, base in sources:
        for _ in range(per_source):
            human = benign_edit(base, rng)
            samples.append(PseudoTextSample(human, 0, name, ["benign"]))
            ai_text, ops = make_pseudo_ai(base, rng)
            ai_text = benign_edit(ai_text, rng)
            samples.append(PseudoTextSample(ai_text, 1, name, ops))
    return samples


if __name__ == "__main__":
    srcs = build_sources()
    print(f"{len(HUMAN_CORPUS)} paragraphs -> {len(srcs)} sources")
    s = generate(per_source=2, seed=0)
    print(f"{len(s)} samples ({sum(x.label == 0 for x in s)} human / "
          f"{sum(x.label == 1 for x in s)} ai)")
    for ex in s[:4]:
        print(f"[{ex.label}] {ex.transforms} :: {ex.text[:90]}...")
