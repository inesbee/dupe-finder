"""
Dupe Finder - matching engine
=============================

Finds cheaper beauty products whose formula is close to a given product.

How it works (in plain words):
1. Ingredient lists are cleaned and turned into a list of standard ingredient
   names ("Aqua/Water/Eau" and "Water" become the same thing).
2. Each product becomes a vector. An ingredient weighs more when:
   - it is rare across all products (TF-IDF idea: water or glycerin are in
     everything, so they say little; niacinamide or bakuchiol say a lot);
   - it appears early in the list (EU/US labelling rules list ingredients in
     decreasing order of concentration, so the first ones matter most).
3. Two products are compared with cosine similarity -> a "% formula match".
4. A Hugging Face sentence-embedding model (all-MiniLM-L6-v2) lets users search
   in natural language ("vitamin C serum for dark spots") instead of typing an
   exact product name. If the model cannot be downloaded, a simpler text search
   is used so the app still works offline.

Data: "Sephora Products and Skincare Reviews" (Kaggle, nadyinky), product_info.csv
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# --------------------------------------------------------------------------
# 1. Cleaning ingredient lists
# --------------------------------------------------------------------------

# Ingredients that buyers recognise and that usually drive a product's effect.
# Used only to label "star ingredients" in the results (not in the score).
STAR_INGREDIENTS = {
    "niacinamide": "Niacinamide",
    "retinol": "Retinol",
    "retinal": "Retinal",
    "bakuchiol": "Bakuchiol",
    "ascorbic acid": "Vitamin C",
    "sodium ascorbyl phosphate": "Vitamin C",
    "ascorbyl glucoside": "Vitamin C",
    "tetrahexyldecyl ascorbate": "Vitamin C",
    "3-o-ethyl ascorbic acid": "Vitamin C",
    "magnesium ascorbyl phosphate": "Vitamin C",
    "sodium hyaluronate": "Hyaluronic acid",
    "hyaluronic acid": "Hyaluronic acid",
    "hydrolyzed hyaluronic acid": "Hyaluronic acid",
    "salicylic acid": "Salicylic acid (BHA)",
    "glycolic acid": "Glycolic acid (AHA)",
    "lactic acid": "Lactic acid (AHA)",
    "mandelic acid": "Mandelic acid (AHA)",
    "azelaic acid": "Azelaic acid",
    "ceramide np": "Ceramides",
    "ceramide ap": "Ceramides",
    "ceramide eop": "Ceramides",
    "squalane": "Squalane",
    "centella asiatica extract": "Centella (cica)",
    "madecassoside": "Centella (cica)",
    "panthenol": "Panthenol (B5)",
    "tocopherol": "Vitamin E",
    "zinc oxide": "Zinc oxide (SPF)",
    "titanium dioxide": "Titanium dioxide",
    "caffeine": "Caffeine",
    "peptides": "Peptides",
    "palmitoyl tripeptide-1": "Peptides",
    "palmitoyl tetrapeptide-7": "Peptides",
    "acetyl hexapeptide-8": "Peptides",
    "copper tripeptide-1": "Peptides",
    "allantoin": "Allantoin",
    "urea": "Urea",
    "shea butter": "Shea butter",
    "butyrospermum parkii butter": "Shea butter",
    "snail secretion filtrate": "Snail mucin",
    "tranexamic acid": "Tranexamic acid",
    "alpha-arbutin": "Alpha-arbutin",
    "arbutin": "Arbutin",
    "kojic acid": "Kojic acid",
    "adenosine": "Adenosine",
    "biotin": "Biotin",
    "keratin": "Keratin",
    "hydrolyzed keratin": "Keratin",
    "argania spinosa kernel oil": "Argan oil",
    "simmondsia chinensis seed oil": "Jojoba oil",
    "rosa canina fruit oil": "Rosehip oil",
}

# Common synonyms written in different ways on labels.
SYNONYMS = {
    "aqua": "water", "eau": "water", "aqua/water": "water", "water/aqua": "water",
    "aqua/water/eau": "water", "water/aqua/eau": "water", "purified water": "water",
    "parfum": "fragrance", "fragrance/parfum": "fragrance", "parfum/fragrance": "fragrance",
    "perfume": "fragrance", "aroma": "fragrance",
    "vitamin e": "tocopherol", "vitamin c": "ascorbic acid",
    "vitamin b3": "niacinamide", "vitamin b5": "panthenol",
    "butyrospermum parkii (shea) butter": "shea butter",
    "alcohol denat": "alcohol denat.",
}

_MAY_CONTAIN = re.compile(r"(may contain|\+/-|\[\+/-\]|peut contenir)", re.I)
_PARENS = re.compile(r"\([^)0-9]*\)")   # "(rice)" removed, "(c6-14 olefin)" kept
_SPACES = re.compile(r"\s+")
_TRAILING = re.compile(r"[\s.*†‡®™°]+$")


def _pick_main_list(raw: str) -> str:
    """The CSV stores ingredients as a Python-style list of strings.

    Most products have one string. Kits/palettes have several
    ("Shade A:", "<list>", "Shade B:", "<list>"); we keep the longest list,
    which is usually the main product.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    try:
        parts = ast.literal_eval(raw)
        if isinstance(parts, str):
            parts = [parts]
    except (ValueError, SyntaxError):
        parts = [raw.strip("[]'\"")]
    parts = [p for p in parts if isinstance(p, str)]
    if not parts:
        return ""
    return max(parts, key=lambda p: p.count(","))


def clean_ingredient(token: str) -> str:
    t = token.lower().strip()
    t = t.replace("*", "").replace("®", "").replace("™", "")
    t = _PARENS.sub(" ", t)            # "oryza sativa (rice) bran" -> "oryza sativa bran"
    t = t.replace("–", "-").replace("—", "-")
    t = _SPACES.sub(" ", t).strip()
    t = _TRAILING.sub("", t).strip()
    if "/" in t and len(t) < 40:        # "water/aqua/eau" == "aqua/water/eau"
        parts = [p.strip() for p in t.split("/") if p.strip()]
        if all(p in ("water", "aqua", "eau", "purified water") for p in parts):
            return "water"
        if all(p in ("fragrance", "parfum", "perfume", "aroma") for p in parts):
            return "fragrance"
    t = SYNONYMS.get(t, t)
    return t


def parse_ingredients(raw: str) -> list[str]:
    """Turn a raw ingredient label into a clean, ordered list."""
    text = _pick_main_list(raw)
    if not text:
        return []
    m = _MAY_CONTAIN.search(text)
    if m:
        text = text[: m.start()]
    # split on commas that are not inside parentheses
    tokens, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            tokens.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    tokens.append("".join(cur))
    out, seen = [], set()
    for tok in tokens:
        c = clean_ingredient(tok)
        if 2 <= len(c) <= 80 and c not in seen and not c.startswith("please "):
            seen.add(c)
            out.append(c)
    return out


# --------------------------------------------------------------------------
# 2. Size parsing (so we can compare price per ounce, not just price)
# --------------------------------------------------------------------------

_OZ = re.compile(r"(\d+(?:\.\d+)?)\s*(?:fl\.?\s*)?oz", re.I)
_ML = re.compile(r"(\d+(?:\.\d+)?)\s*ml", re.I)
_G = re.compile(r"(\d+(?:\.\d+)?)\s*g\b", re.I)


def size_in_oz(size: str) -> float | None:
    if not isinstance(size, str):
        return None
    for rx, factor in ((_OZ, 1.0), (_ML, 1 / 29.5735), (_G, 1 / 28.3495)):
        m = rx.search(size)
        if m:
            v = float(m.group(1)) * factor
            return v if v > 0 else None
    return None


# --------------------------------------------------------------------------
# 3. The engine
# --------------------------------------------------------------------------

@dataclass
class Match:
    row: pd.Series
    score: float               # 0-100 formula match
    shared_top: list[str]      # shared ingredients, most important first
    shared_stars: list[str]    # shared "star" actives, human-readable
    n_shared: int
    top10_shared: int          # how many of the target's first 10 ingredients it also has

    @property
    def verdict(self) -> str:
        if self.score >= 60:
            return "Near-twin formula"
        if self.score >= 40:
            return "Very close formula"
        if self.score >= 25:
            return "Similar formula"
        return "Loose match"


class DupeFinder:
    EXCLUDED_PRIMARY = {"Mini Size", "Gifts", "Tools & Brushes"}
    EXCLUDED_WORDS = re.compile(r"\b(?:mini|travel size|set|kit|duo|trio|value)\b", re.I)
    STAR_BOOST = 1.5   # active ingredients count a bit more: they drive what a product does

    def __init__(self, csv_path: str = "product_info.csv", use_hf_model: bool = True,
                 hf_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        df = pd.read_csv(csv_path)
        df["ingredient_list"] = df["ingredients"].map(parse_ingredients)
        df = df[df["ingredient_list"].map(len) >= 4].copy()
        df = df[~df["primary_category"].isin(self.EXCLUDED_PRIMARY)]
        df = df[~df["product_name"].str.contains(self.EXCLUDED_WORDS, na=False)]
        df["category"] = df["tertiary_category"].fillna(df["secondary_category"]).fillna(
            df["primary_category"])
        df["size_oz"] = df["size"].map(size_in_oz)
        df["price_per_oz"] = np.where(df["size_oz"].notna(), df["price_usd"] / df["size_oz"], np.nan)
        df["label"] = df["brand_name"] + " - " + df["product_name"]
        self.df = df.reset_index(drop=True)
        self._build_formula_matrix()
        self._build_search(use_hf_model, hf_model_name)

    # ---- formula vectors -------------------------------------------------
    def _build_formula_matrix(self):
        lists = self.df["ingredient_list"].tolist()
        vocab: dict[str, int] = {}
        doc_freq: dict[int, int] = {}
        for ings in lists:
            for ing in ings:
                j = vocab.setdefault(ing, len(vocab))
                doc_freq[j] = doc_freq.get(j, 0) + 1
        n = len(lists)
        idf = np.zeros(len(vocab))
        for j, dfreq in doc_freq.items():
            idf[j] = math.log((1 + n) / (1 + dfreq)) + 1.0
        self.vocab, self.idf = vocab, idf
        self.X = sparse.vstack([self._vector_for(ings) for ings in lists]).tocsr()

    def _weight(self, ing: str, rank: int) -> float:
        """Importance of one ingredient in one product."""
        j = self.vocab[ing]
        position_weight = 1.0 / math.sqrt(1 + rank)   # earlier in the list = more concentrated
        boost = self.STAR_BOOST if ing in STAR_INGREDIENTS else 1.0
        return self.idf[j] * position_weight * boost

    def _vector_for(self, ingredients: list[str]):
        cols, vals = [], []
        for rank, ing in enumerate(ingredients):
            if ing in self.vocab:
                cols.append(self.vocab[ing]); vals.append(self._weight(ing, rank))
        v = sparse.csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, len(self.vocab)))
        return normalize(v)

    # ---- product search (Hugging Face model) -----------------------------
    def _search_text(self) -> list[str]:
        hl = self.df["highlights"].fillna("").str.replace(r"[\[\]']", "", regex=True)
        return (self.df["brand_name"] + " " + self.df["product_name"] + ". " +
                self.df["category"].fillna("") + ". " + hl).tolist()

    def _build_search(self, use_hf_model: bool, model_name: str):
        texts = self._search_text()
        self.search_mode = "keyword"
        self.embedder = None
        if use_hf_model:
            try:
                from sentence_transformers import SentenceTransformer
                self.embedder = SentenceTransformer(model_name)
                self.emb = self.embedder.encode(texts, batch_size=128, show_progress_bar=True,
                                                normalize_embeddings=True)
                self.search_mode = "semantic (Hugging Face: " + model_name.split("/")[-1] + ")"
            except Exception as e:  # no internet / library missing -> fallback
                print(f"[DupeFinder] Hugging Face model unavailable ({e.__class__.__name__}); "
                      "using keyword search instead.")
                self.embedder = None
        # keyword search (always built; also used for exact-name matching)
        names = (self.df["brand_name"] + " " + self.df["product_name"]).str.lower().tolist()
        self.tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True)
        self.tfidf_word = TfidfVectorizer(analyzer="word", token_pattern=r"[\w\-\.\+]+",
                                          sublinear_tf=True)
        self.Tc = self.tfidf_char.fit_transform(names)
        self.Tw = self.tfidf_word.fit_transform([t.lower() for t in texts])
        self._name_words = [set(re.findall(r"[\w\-\+]+", n)) for n in names]

    def search_products(self, query: str, k: int = 8) -> list[int]:
        """Return row indices of the products that best match a free-text query."""
        query = (query or "").strip()
        if not query:
            return []
        exact = self.df.index[self.df["label"].str.lower() == query.lower()].tolist()
        if exact:
            return exact[:1]
        q = query.lower()
        q_words = set(re.findall(r"[\w\-\+]+", q))
        overlap = np.array([len(q_words & w) / max(1, len(q_words)) for w in self._name_words])
        kw = 0.35 * (self.Tc @ self.tfidf_char.transform([q]).T).toarray().ravel() + \
             0.35 * (self.Tw @ self.tfidf_word.transform([q]).T).toarray().ravel() + \
             0.30 * overlap
        if self.embedder is not None:
            q = self.embedder.encode([query], normalize_embeddings=True)[0]
            sem = self.emb @ q
            scores = 0.6 * sem + 0.4 * kw      # semantic meaning + spelling of names
        else:
            scores = kw
        return list(np.argsort(-scores)[:k])

    # ---- dupes -------------------------------------------------------------
    def _make_match(self, target_ings: list[str], j: int, score: float) -> Match:
        row = self.df.loc[j]
        b = set(row["ingredient_list"])
        shared = [x for x in target_ings if x in b]
        rank = {x: r for r, x in enumerate(target_ings)}
        shared_sorted = sorted(shared, key=lambda x: -self._weight(x, rank[x]))
        stars, seen = [], set()
        for x in shared:
            name = STAR_INGREDIENTS.get(x)
            if name and name not in seen:
                seen.add(name); stars.append(name)
        top10 = sum(1 for x in target_ings[:10] if x in b)
        return Match(row=row, score=score * 100, shared_top=shared_sorted[:6],
                     shared_stars=stars, n_shared=len(shared), top10_shared=top10)

    def _rank(self, target_vec, target_ings, mask, k) -> list[Match]:
        sims = (self.X @ target_vec.T).toarray().ravel()
        mask = mask & (sims < 0.999)   # identical formula = same product re-listed
        cand = np.where(mask)[0]
        out, picked = [], []
        for j in cand[np.argsort(-sims[cand])]:
            # skip near-copies of a result we already show (e.g. same serum, two names)
            if any((self.X[j] @ self.X[p].T).toarray()[0, 0] > 0.97 for p in picked):
                continue
            picked.append(j)
            out.append(self._make_match(target_ings, j, float(sims[j])))
            if len(out) == k:
                break
        return out

    def find_dupes(self, idx: int, k: int = 5, cheaper_only: bool = True,
                   same_category: bool = True, other_brands: bool = True) -> list[Match]:
        target = self.df.loc[idx]
        mask = np.ones(len(self.df), dtype=bool)
        mask[idx] = False
        if same_category:
            mask &= (self.df["category"] == target["category"]).values
        if cheaper_only:
            mask &= (self.df["price_usd"] < target["price_usd"]).values
        if other_brands:
            mask &= (self.df["brand_name"] != target["brand_name"]).values
        return self._rank(self.X[idx], target["ingredient_list"], mask, k)

    def find_from_ingredients(self, raw_text: str, k: int = 5, max_price: float | None = None,
                              category: str | None = None) -> tuple[list[Match], list[str]]:
        """Paste any ingredient list (e.g. from a product not sold at Sephora)."""
        ings = parse_ingredients(str([raw_text]))
        if not ings:
            return [], []
        mask = np.ones(len(self.df), dtype=bool)
        if max_price:
            mask &= (self.df["price_usd"] <= max_price).values
        if category and category != "All categories":
            mask &= (self.df["category"] == category).values
        return self._rank(self._vector_for(ings), ings, mask, k), ings
