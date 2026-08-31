"""
analisis_lab5.py
================
Laboratorio 5 – Clasificación de tweets sobre desastres
Universidad del Valle de Guatemala – CC3084 Data Science – Semestre II 2026

Uso:
    python analisis_lab5.py --eda            # Ejecuta el pipeline EDA completo
    python analisis_lab5.py --validate-eda   # Valida los artefactos generados

Semilla global: 42
"""

import argparse
import html
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    warnings.warn("wordcloud no instalado; nubes de palabras deshabilitadas.")

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(".")
OUTPUT_DIR = Path("outputs/eda")
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = Path("outputs/models")
SENTIMENT_DIR = Path("outputs/sentiment")
FIG_DPI = 150

NEGATIONS = frozenset({
    "no", "not", "nor", "never", "neither",
    "nobody", "nothing", "nowhere",
    "hardly", "scarcely", "barely",
    "dont", "doesnt", "didnt",
    "wont", "wouldnt", "shouldnt", "couldnt",
    "hasnt", "havent", "hadnt",
    "isnt", "arent", "wasnt", "werent",
    "aint", "cant", "cannot",
    "don", "doesn", "didn", "won", "wouldn", "shouldn", "couldn",
    "hasn", "haven", "hadn", "isn", "aren", "wasn", "weren", "ain",
})

STOPWORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

CLASS_LABELS = {0: "No desastre", 1: "Desastre"}
CLASS_COLORS = {0: "#4C72B0", 1: "#DD8452"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_sentiment_dirs():
    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  1. CARGA Y VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def load_datasets(data_dir=None):
    d = Path(data_dir) if data_dir else DATA_DIR
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv")
    submission = pd.read_csv(d / "sample_submission.csv")
    return train, test, submission


def validate_schemas(train, test, submission):
    errors = []

    if train.shape != (7613, 5):
        errors.append(f"train shape {train.shape} != (7613, 5)")
    if list(train.columns) != ["id", "keyword", "location", "text", "target"]:
        errors.append(f"train columns unexpected: {list(train.columns)}")
    if train["id"].nunique() != len(train):
        errors.append("train IDs not unique")
    if set(train["target"].unique()) != {0, 1}:
        errors.append("train target not {0,1}")
    if train["target"].value_counts().get(0, 0) != 4342:
        errors.append("train target=0 count mismatch")
    if train["target"].value_counts().get(1, 0) != 3271:
        errors.append("train target=1 count mismatch")
    if train["keyword"].isna().sum() != 61:
        errors.append("train keyword missing count mismatch")
    if train["location"].isna().sum() != 2533:
        errors.append("train location missing count mismatch")
    if train["text"].isna().sum() != 0:
        errors.append("train text has missing values")

    if test.shape != (3263, 4):
        errors.append(f"test shape {test.shape} != (3263, 4)")
    if "target" in test.columns:
        errors.append("test should not contain target")
    if test["id"].nunique() != len(test):
        errors.append("test IDs not unique")

    if submission.shape != (3263, 2):
        errors.append(f"submission shape {submission.shape} != (3263, 2)")
    if list(submission.columns) != ["id", "target"]:
        errors.append("submission columns mismatch")
    if not (submission["id"].values == test["id"].values).all():
        errors.append("submission IDs don't match test IDs")

    overlap = set(train["id"]).intersection(set(test["id"]))
    if len(overlap) > 0:
        errors.append(f"ID overlap between train and test: {len(overlap)}")

    return {"valid": len(errors) == 0, "errors": errors}


# ═══════════════════════════════════════════════════════════════════════════
#  2. LIMPIEZA DE TEXTO
# ═══════════════════════════════════════════════════════════════════════════

def clean_for_analysis(text: str) -> str:
    """Limpieza para análisis léxico.

    - Decodifica entidades HTML.
    - Convierte a minúsculas.
    - Elimina URLs.
    - Retira '#' conservando la palabra (e.g. #earthquake → earthquake).
    - Elimina menciones (@usuario) sin dañar palabras vecinas.
    - Colapsa apóstrofos (don't → dont) para preservar negaciones.
    - Elimina puntuación restante, conserva letras y dígitos.
    - Conserva números como 911 (relevante para desastres).
    - Normaliza espacios.
    """
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords_text(text: str) -> str:
    """Elimina stopwords preservando negaciones."""
    return " ".join(t for t in text.split() if t not in STOPWORDS)


def clean_for_model(text: str) -> str:
    """Limpieza ligera para uso posterior con TF-IDF.

    Menos agresiva: conserva negaciones, hashtags como palabras,
    y no elimina stopwords (el vectorizador se encargará).
    """
    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decode_keyword(kw):
    """Decodifica keywords URL-encoded (e.g. body%20bags → body bags)."""
    if pd.isna(kw):
        return kw
    return unquote(str(kw))


def preprocess_dataframe(df):
    """Agrega columnas de texto limpio al DataFrame."""
    df = df.copy()
    df["text_clean"] = df["text"].map(clean_for_analysis)
    df["text_no_stop"] = df["text_clean"].map(remove_stopwords_text)
    df["text_model"] = df["text"].map(clean_for_model)
    df["keyword_decoded"] = df["keyword"].map(decode_keyword)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  3. FEATURES DERIVADOS DEL TEXTO
# ═══════════════════════════════════════════════════════════════════════════

def add_text_features(df):
    """Agrega conteos derivados del texto original."""
    df = df.copy()
    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    df["url_count"] = df["text"].str.count(r"https?://\S+|www\.\S+")
    df["mention_count"] = df["text"].str.count(r"@\w+")
    df["hashtag_count"] = df["text"].str.count(r"#\w+")
    df["number_count"] = df["text"].str.count(r"\b\d+\b")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  4. AUDITORÍA DE DUPLICADOS
# ═══════════════════════════════════════════════════════════════════════════

def audit_duplicates(df, method="exact"):
    """Analiza textos repetidos y etiquetas contradictorias.

    method: "exact" (texto tal cual) o "normalized" (strip + lower).
    La clave normalizada se usa para agrupar particiones;
    la exacta, para reportar los duplicados literales.
    """
    if method == "normalized":
        key = df["text"].str.strip().str.lower()
    else:
        key = df["text"]

    df_keyed = df.assign(_key=key)
    group_sizes = df_keyed.groupby("_key").size()
    dup_keys = group_sizes[group_sizes > 1].index
    dup_df = df_keyed[df_keyed["_key"].isin(dup_keys)]

    n_dup_groups = len(dup_keys)
    n_dup_rows = len(dup_df)

    if n_dup_groups == 0:
        return {
            "method": method,
            "n_duplicate_groups": 0, "n_duplicate_rows": 0,
            "n_same_label_groups": 0, "n_same_label_rows": 0,
            "n_contradictory_groups": 0, "n_contradictory_rows": 0,
            "contradictions": pd.DataFrame(),
            "duplicate_summary": pd.DataFrame(),
        }

    label_nunique = dup_df.groupby("_key")["target"].nunique()
    contra_keys = label_nunique[label_nunique > 1].index
    same_keys = label_nunique[label_nunique == 1].index

    contra_df = dup_df[dup_df["_key"].isin(contra_keys)].drop(columns="_key")
    same_df = dup_df[dup_df["_key"].isin(same_keys)]

    summary_rows = []
    for k in dup_keys:
        grp = dup_df[dup_df["_key"] == k]
        summary_rows.append({
            "text_preview": grp["text"].iloc[0][:80],
            "n_copies": len(grp),
            "targets": sorted(grp["target"].unique().tolist()),
            "contradictory": grp["target"].nunique() > 1,
        })
    summary = pd.DataFrame(summary_rows)

    return {
        "method": method,
        "n_duplicate_groups": n_dup_groups,
        "n_duplicate_rows": n_dup_rows,
        "n_same_label_groups": len(same_keys),
        "n_same_label_rows": len(same_df),
        "n_contradictory_groups": len(contra_keys),
        "n_contradictory_rows": len(contra_df),
        "contradictions": contra_df,
        "duplicate_summary": summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. EDA – FIGURAS
# ═══════════════════════════════════════════════════════════════════════════

def plot_target_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = df["target"].value_counts().sort_index()

    bars = axes[0].bar(
        [CLASS_LABELS[0], CLASS_LABELS[1]], counts.values,
        color=[CLASS_COLORS[0], CLASS_COLORS[1]], edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                     f"{val:,}", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Número de tweets")
    axes[0].set_title("Distribución de target")
    axes[0].set_ylim(0, max(counts.values) * 1.15)

    axes[1].pie(
        counts.values,
        labels=[f"{CLASS_LABELS[i]}\n({counts[i]:,}, {counts[i]/len(df)*100:.1f}%)" for i in [0, 1]],
        colors=[CLASS_COLORS[0], CLASS_COLORS[1]],
        startangle=90, autopct="", wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Proporción de clases")

    fig.suptitle("Variable objetivo: ¿el tweet describe un desastre real?", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "target_distribution.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_missing_values(df):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(missing.index, missing.values, color="#7A9CC6", edgecolor="white")
    for bar, val in zip(bars, missing.values):
        pct = val / len(df) * 100
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,} ({pct:.1f}%)", va="center", fontsize=10)
    ax.set_xlabel("Valores faltantes")
    ax.set_title("Valores faltantes por columna (train.csv)")
    ax.set_xlim(0, max(missing.values) * 1.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "missing_values.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_text_lengths(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for target_val in [0, 1]:
        subset = df[df["target"] == target_val]
        axes[0].hist(subset["char_count"], bins=40, alpha=0.6,
                     label=CLASS_LABELS[target_val], color=CLASS_COLORS[target_val])
        axes[1].hist(subset["word_count"], bins=30, alpha=0.6,
                     label=CLASS_LABELS[target_val], color=CLASS_COLORS[target_val])

    axes[0].set_xlabel("Caracteres por tweet")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title("Longitud en caracteres por clase")
    axes[0].legend()

    axes[1].set_xlabel("Palabras por tweet")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Longitud en palabras por clase")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "text_length_by_target.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_feature_counts(df):
    features = ["url_count", "mention_count", "hashtag_count", "number_count"]
    labels = ["URLs", "Menciones (@)", "Hashtags (#)", "Números"]

    means = {}
    for t in [0, 1]:
        subset = df[df["target"] == t]
        means[t] = [subset[f].mean() for f in features]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, means[0], width, label=CLASS_LABELS[0],
           color=CLASS_COLORS[0], edgecolor="white")
    ax.bar(x + width / 2, means[1], width, label=CLASS_LABELS[1],
           color=CLASS_COLORS[1], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Promedio por tweet")
    ax.set_title("Elementos de texto promedio por clase")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_counts_by_target.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_keyword_frequency(df, top_n=30):
    kw = df["keyword_decoded"].dropna().value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(kw.index[::-1], kw.values[::-1], color="#7A9CC6", edgecolor="white")
    ax.set_xlabel("Frecuencia")
    ax.set_title(f"Top {top_n} keywords más frecuentes")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "keyword_top30.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_disaster_rate_by_keyword(df, min_obs=10):
    kw_stats = df.groupby("keyword_decoded").agg(
        n=("target", "size"),
        disaster_rate=("target", "mean"),
    ).reset_index()
    kw_stats = kw_stats[kw_stats["n"] >= min_obs].sort_values("disaster_rate")

    top_safe = kw_stats.head(15)
    top_danger = kw_stats.tail(15)
    combined = pd.concat([top_safe, top_danger])

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ["#4C72B0" if r < 0.5 else "#DD8452" for r in combined["disaster_rate"]]
    ax.barh(combined["keyword_decoded"], combined["disaster_rate"], color=colors, edgecolor="white")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Proporción de desastre")
    ax.set_title(f"Proporción de desastre por keyword (mín. {min_obs} obs.)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "disaster_rate_by_keyword.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    kw_stats.to_csv(TABLES_DIR / "disaster_rate_by_keyword.csv", index=False)
    return fig


def plot_top_locations(df, top_n=20):
    loc = df["location"].dropna()
    loc_clean = loc.str.strip().str.lower()
    top = loc_clean.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index[::-1], top.values[::-1], color="#7A9CC6", edgecolor="white")
    ax.set_xlabel("Frecuencia")
    ax.set_title(f"Top {top_n} ubicaciones (alta cardinalidad: {df['location'].nunique()} únicas)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_locations.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  6. N-GRAMAS
# ═══════════════════════════════════════════════════════════════════════════

def compute_ngrams(texts, n=1):
    """Calcula n-gramas a partir de una serie de textos tokenizados por espacios."""
    counter = Counter()
    for text in texts:
        tokens = text.split()
        if n == 1:
            counter.update(tokens)
        else:
            counter.update(zip(*[tokens[i:] for i in range(n)]))
    if n > 1:
        counter = Counter({" ".join(k): v for k, v in counter.items()})
    return counter


def ngram_table_by_class(df, n=1, top_k=30):
    """Tabla de frecuencias de n-gramas por clase con log-odds ratio."""
    counters = {}
    totals = {}
    for t in [0, 1]:
        texts = df[df["target"] == t]["text_no_stop"]
        c = compute_ngrams(texts, n)
        counters[t] = c
        totals[t] = sum(c.values())

    all_ngrams = set(counters[0].keys()) | set(counters[1].keys())
    V = len(all_ngrams)
    alpha = 1.0

    rows = []
    for ng in all_ngrams:
        f0 = counters[0].get(ng, 0)
        f1 = counters[1].get(ng, 0)
        p0 = (f0 + alpha) / (totals[0] + alpha * V)
        p1 = (f1 + alpha) / (totals[1] + alpha * V)
        log_odds = np.log(p1 / p0)
        rows.append({
            "ngram": ng,
            "freq_no_desastre": f0,
            "freq_desastre": f1,
            "freq_total": f0 + f1,
            "prob_no_desastre": round(f0 / totals[0], 6) if totals[0] > 0 else 0,
            "prob_desastre": round(f1 / totals[1], 6) if totals[1] > 0 else 0,
            "log_odds": round(log_odds, 4),
        })

    table = pd.DataFrame(rows)
    table["abs_log_odds"] = table["log_odds"].abs()
    table = table.sort_values("freq_total", ascending=False)
    return table


def plot_ngrams_by_class(df, n=1, top_k=20):
    """Gráfica de los top n-gramas para cada clase."""
    names = {1: "Unigramas", 2: "Bigramas", 3: "Trigramas"}
    name = names.get(n, f"{n}-gramas")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for idx, target_val in enumerate([0, 1]):
        texts = df[df["target"] == target_val]["text_no_stop"]
        counter = compute_ngrams(texts, n)
        top = counter.most_common(top_k)
        ngrams = [x[0] for x in top][::-1]
        freqs = [x[1] for x in top][::-1]

        axes[idx].barh(ngrams, freqs, color=CLASS_COLORS[target_val], edgecolor="white")
        axes[idx].set_xlabel("Frecuencia")
        axes[idx].set_title(f"{name} – {CLASS_LABELS[target_val]}")

    fig.suptitle(f"Top {top_k} {name.lower()} por clase", fontsize=13, y=1.02)
    fig.tight_layout()
    fname = f"{name.lower()}_by_class.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def save_ngram_tables(df):
    """Genera y guarda tablas de n-gramas."""
    tables = {}
    for n, name in [(1, "unigrams"), (2, "bigrams"), (3, "trigrams")]:
        table = ngram_table_by_class(df, n=n)
        table.to_csv(TABLES_DIR / f"{name}_by_class.csv", index=False)
        tables[name] = table
        print(f"  {name}: {len(table)} {name} únicos guardados.")
    return tables


def show_distinctive_ngrams(table, top_k=15):
    """Muestra los n-gramas más distintivos por clase."""
    min_freq = 5
    filtered = table[table["freq_total"] >= min_freq].copy()

    disaster_terms = filtered.nlargest(top_k, "log_odds")[
        ["ngram", "freq_desastre", "freq_no_desastre", "log_odds"]
    ]
    no_disaster_terms = filtered.nsmallest(top_k, "log_odds")[
        ["ngram", "freq_desastre", "freq_no_desastre", "log_odds"]
    ]
    ambiguous = filtered.nsmallest(top_k, "abs_log_odds")[
        ["ngram", "freq_desastre", "freq_no_desastre", "log_odds"]
    ]

    return {
        "disaster": disaster_terms,
        "no_disaster": no_disaster_terms,
        "ambiguous": ambiguous,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  7. NUBES DE PALABRAS
# ═══════════════════════════════════════════════════════════════════════════

def generate_wordcloud_for_class(df, target_val, colormap=None):
    """Genera y guarda una nube de palabras para una clase."""
    if not HAS_WORDCLOUD:
        warnings.warn("wordcloud no disponible.")
        return None

    texts = df[df["target"] == target_val]["text_no_stop"]
    all_text = " ".join(texts)

    if colormap is None:
        colormap = "Blues" if target_val == 0 else "Oranges"

    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        max_words=120,
        random_state=SEED,
        colormap=colormap,
        collocations=False,
    )
    wc.generate(all_text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Nube de palabras – {CLASS_LABELS[target_val]}", fontsize=13)
    fig.tight_layout()
    fname = f"wordcloud_target{target_val}.png"
    fig.savefig(OUTPUT_DIR / fname, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


def generate_all_wordclouds(df):
    """Genera nubes de palabras para ambas clases."""
    generate_wordcloud_for_class(df, 0)
    generate_wordcloud_for_class(df, 1)


# ═══════════════════════════════════════════════════════════════════════════
#  8. PIPELINE EDA COMPLETO
# ═══════════════════════════════════════════════════════════════════════════

def run_eda():
    """Ejecuta el pipeline EDA completo."""
    ensure_dirs()
    print("=" * 60)
    print("LABORATORIO 5 – Pipeline EDA")
    print("=" * 60)

    # 1. Carga
    print("\n[1/7] Cargando datos...")
    train, test, submission = load_datasets()
    report = validate_schemas(train, test, submission)
    if not report["valid"]:
        print("ERRORES de validación:")
        for e in report["errors"]:
            print(f"  X {e}")
        sys.exit(1)
    print("  Esquemas validados correctamente.")

    # 2. Preprocesamiento
    print("\n[2/7] Preprocesando textos...")
    train = preprocess_dataframe(train)
    train = add_text_features(train)
    empty_after_clean = (train["text_clean"] == "").sum()
    print(f"  Textos vacíos tras limpieza: {empty_after_clean}")
    print(f"  Ejemplo: '{train['text'].iloc[0]}'")
    print(f"        -> '{train['text_clean'].iloc[0]}'")
    print(f"  Sin SW -> '{train['text_no_stop'].iloc[0]}'")

    # 3. Duplicados
    print("\n[3/7] Auditando duplicados...")
    exact_report = audit_duplicates(train, method="exact")
    norm_report = audit_duplicates(train, method="normalized")
    print("  Duplicados exactos:")
    print(f"    Grupos: {exact_report['n_duplicate_groups']}, Filas: {exact_report['n_duplicate_rows']}")
    print(f"    Contradicciones: {exact_report['n_contradictory_groups']} grupos / {exact_report['n_contradictory_rows']} filas")
    print("  Duplicados normalizados (strip+lower):")
    print(f"    Grupos: {norm_report['n_duplicate_groups']}, Filas: {norm_report['n_duplicate_rows']}")
    print(f"    Contradicciones: {norm_report['n_contradictory_groups']} grupos / {norm_report['n_contradictory_rows']} filas")
    if len(norm_report["duplicate_summary"]) > 0:
        norm_report["duplicate_summary"].to_csv(TABLES_DIR / "duplicate_audit.csv", index=False)
    if len(norm_report["contradictions"]) > 0:
        norm_report["contradictions"].to_csv(TABLES_DIR / "contradictory_labels.csv", index=False)

    # 4. Figuras EDA
    print("\n[4/7] Generando figuras EDA...")
    plot_target_distribution(train)
    print("  target_distribution.png")
    plot_missing_values(train)
    print("  missing_values.png")
    plot_text_lengths(train)
    print("  text_length_by_target.png")
    plot_feature_counts(train)
    print("  feature_counts_by_target.png")
    plot_keyword_frequency(train)
    print("  keyword_top30.png")
    plot_disaster_rate_by_keyword(train)
    print("  disaster_rate_by_keyword.png")
    plot_top_locations(train)
    print("  top_locations.png")

    # 5. N-gramas
    print("\n[5/7] Calculando n-gramas...")
    tables = save_ngram_tables(train)

    # 6. Figuras de n-gramas
    print("\n[6/7] Generando figuras de n-gramas...")
    for n in [1, 2, 3]:
        plot_ngrams_by_class(train, n=n, top_k=20)
    print("  unigramas_by_class.png")
    print("  bigramas_by_class.png")
    print("  trigramas_by_class.png")

    # 7. Nubes de palabras
    print("\n[7/7] Generando nubes de palabras...")
    generate_all_wordclouds(train)
    print("  wordcloud_target0.png")
    print("  wordcloud_target1.png")

    # Manifiesto
    artifacts = sorted(str(p) for p in OUTPUT_DIR.rglob("*") if p.is_file())
    manifest = {"seed": SEED, "artifacts": artifacts}
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"EDA completo. {len(artifacts)} artefactos en {OUTPUT_DIR}/")
    print("=" * 60)
    return train, tables


# ═══════════════════════════════════════════════════════════════════════════
#  9. VALIDACIÓN
# ═══════════════════════════════════════════════════════════════════════════

def validate_eda():
    """Valida que todos los artefactos EDA existan y sean válidos."""
    ensure_dirs()
    errors = []

    # Esquemas
    try:
        train, test, submission = load_datasets()
        report = validate_schemas(train, test, submission)
        if not report["valid"]:
            errors.extend(report["errors"])
    except Exception as e:
        errors.append(f"Error cargando datos: {e}")

    # Figuras
    expected_figs = [
        "target_distribution.png", "missing_values.png",
        "text_length_by_target.png", "feature_counts_by_target.png",
        "keyword_top30.png", "disaster_rate_by_keyword.png",
        "top_locations.png",
        "unigramas_by_class.png", "bigramas_by_class.png", "trigramas_by_class.png",
        "wordcloud_target0.png", "wordcloud_target1.png",
    ]
    for fname in expected_figs:
        fpath = OUTPUT_DIR / fname
        if not fpath.exists():
            errors.append(f"Figura faltante: {fname}")
        elif fpath.stat().st_size < 1000:
            errors.append(f"Figura sospechosamente pequeña: {fname}")

    # Tablas
    expected_tables = [
        "unigrams_by_class.csv", "bigrams_by_class.csv", "trigrams_by_class.csv",
        "disaster_rate_by_keyword.csv",
    ]
    for fname in expected_tables:
        fpath = TABLES_DIR / fname
        if not fpath.exists():
            errors.append(f"Tabla faltante: {fname}")
        else:
            t = pd.read_csv(fpath)
            if len(t) == 0:
                errors.append(f"Tabla vacía: {fname}")

    # Ausencia de etiqueta proxy
    try:
        nb_path = DATA_DIR / "lab5.ipynb"
        if nb_path.exists():
            content = nb_path.read_text(encoding="utf-8")
            kp_count = content.count("keyword_present")
            if kp_count > 2:
                errors.append(f"El notebook contiene 'keyword_present' {kp_count} veces (posible etiqueta proxy activa)")
            if "0.98" in content and "accuracy" in content.lower():
                errors.append("El notebook contiene la accuracy 0.98 de la etiqueta proxy")
    except Exception:
        pass

    # Rutas absolutas
    for pyfile in [DATA_DIR / "analisis_lab5.py"]:
        if pyfile.exists():
            text = pyfile.read_text(encoding="utf-8")
            if re.search(r"[A-Z]:\\Users", text):
                errors.append(f"Ruta absoluta local en {pyfile.name}")

    # Marcadores pendientes (excluye el propio validador)
    marker = "PEND" + "IENTE"
    for pyfile in DATA_DIR.glob("*.py"):
        if pyfile.name == "analisis_lab5.py":
            continue
        text = pyfile.read_text(encoding="utf-8")
        if marker in text:
            errors.append(f"Marcador {marker} en {pyfile.name}")

    print("=" * 60)
    print("VALIDACIÓN EDA")
    print("=" * 60)
    if errors:
        print(f"\n{len(errors)} problemas encontrados:")
        for e in errors:
            print(f"  X {e}")
    else:
        print("\nTodos los controles pasaron correctamente.")
    print("=" * 60)
    return errors


# ═══════════════════════════════════════════════════════════════════════════
#  10. MODELADO
# ═══════════════════════════════════════════════════════════════════════════

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedGroupKFold, cross_val_predict,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer


class TextCleaner(BaseEstimator, TransformerMixin):
    """Limpieza de texto integrable en Pipeline de sklearn."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_for_model(t) for t in X]


def create_group_stratified_split(X, y, groups, test_size_folds=5):
    """Division dev/holdout estratificada por grupos (~20% holdout)."""
    sgkf = StratifiedGroupKFold(
        n_splits=test_size_folds, shuffle=True, random_state=SEED,
    )
    dev_idx, hold_idx = next(sgkf.split(X, y, groups))

    shared = set(groups[dev_idx]) & set(groups[hold_idx])
    return {
        "dev_idx": dev_idx, "holdout_idx": hold_idx,
        "X_dev": X[dev_idx], "y_dev": y[dev_idx],
        "groups_dev": groups[dev_idx],
        "X_hold": X[hold_idx], "y_hold": y[hold_idx],
        "groups_hold": groups[hold_idx],
        "shared_groups": len(shared),
    }


def get_model_configs():
    """Modelos y grids de busqueda."""
    base_tfidf = dict(min_df=2, max_df=0.95, sublinear_tf=True)
    return {
        "Dummy": (
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), **base_tfidf)),
                ("clf", DummyClassifier(strategy="most_frequent",
                                        random_state=SEED)),
            ]),
            {},
        ),
        "LogReg": (
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), **base_tfidf)),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED,
                                           class_weight="balanced")),
            ]),
            {"tfidf__max_df": [0.9, 0.95], "clf__C": [0.5, 1.0, 5.0]},
        ),
        "ComplementNB": (
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), **base_tfidf)),
                ("clf", ComplementNB()),
            ]),
            {"tfidf__max_df": [0.9, 0.95], "clf__alpha": [0.5, 1.0, 2.0]},
        ),
        "LinearSVC": (
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), **base_tfidf)),
                ("clf", CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, random_state=SEED,
                              class_weight="balanced"),
                    cv=3,
                )),
            ]),
            {
                "tfidf__max_df": [0.9, 0.95],
                "clf__estimator__C": [0.5, 1.0, 5.0],
            },
        ),
        "LogReg_char": (
            Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5),
                    min_df=2, max_df=0.95, sublinear_tf=True,
                )),
                ("clf", LogisticRegression(max_iter=1000, random_state=SEED,
                                           class_weight="balanced")),
            ]),
            {"clf__C": [0.5, 1.0, 5.0]},
        ),
    }


def evaluate_holdout(pipeline, X_hold, y_hold, model_name):
    """Evaluacion unica en holdout. Guarda figuras y metricas."""
    y_pred = pipeline.predict(X_hold)

    metrics = {
        "f1_disaster": round(f1_score(y_hold, y_pred), 4),
        "precision": round(precision_score(y_hold, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_hold, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_hold, y_pred), 4),
        "f1_macro": round(f1_score(y_hold, y_pred, average="macro"), 4),
    }
    result = {"y_pred": y_pred, "metrics": metrics}

    try:
        y_proba = pipeline.predict_proba(X_hold)[:, 1]
        score_tipo = "probabilidad"
    except AttributeError:
        y_proba = pipeline.decision_function(X_hold)
        score_tipo = "margen"
    metrics["pr_auc"] = round(average_precision_score(y_hold, y_proba), 4)
    metrics["roc_auc"] = round(roc_auc_score(y_hold, y_proba), 4)
    result["y_proba"] = y_proba
    result["score_tipo"] = score_tipo

    with open(MODELS_DIR / "holdout_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_hold, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No desastre", "Desastre"])
    ax.set_yticklabels(["No desastre", "Desastre"])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusion - {model_name} (holdout)")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close(fig)

    # PR + ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PrecisionRecallDisplay.from_predictions(y_hold, y_proba, ax=axes[0],
                                            name=model_name)
    axes[0].set_title("Curva Precision-Recall")
    RocCurveDisplay.from_predictions(y_hold, y_proba, ax=axes[1],
                                     name=model_name)
    axes[1].set_title("Curva ROC")
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "pr_roc_curves.png", dpi=FIG_DPI,
                bbox_inches="tight")
    plt.close(fig)

    return result


def clasificar_tweet(texto, pipeline=None):
    """Clasifica un tweet como desastre o no.

    Recibe texto sin procesar. Retorna clase, interpretacion y score.
    Para LinearSVC calibrado el score es probabilidad;
    si se usa decision_function directa, es margen.
    """
    if not isinstance(texto, str) or not texto.strip():
        return {"clase": None, "interpretacion": "Entrada invalida",
                "score": None, "score_tipo": None}

    if pipeline is None:
        import joblib
        pipeline = joblib.load(MODELS_DIR / "best_pipeline.joblib")

    prediction = int(pipeline.predict([texto])[0])

    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        score = float(pipeline.predict_proba([texto])[0, 1])
        score_tipo = "probabilidad"
    elif hasattr(clf, "decision_function"):
        score = float(pipeline.decision_function([texto])[0])
        score_tipo = "margen"
    else:
        score = float(prediction)
        score_tipo = "prediccion"

    return {
        "clase": prediction,
        "interpretacion": CLASS_LABELS[prediction],
        "score": round(score, 4),
        "score_tipo": score_tipo,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  11. PIPELINES DE EJECUCION
# ═══════════════════════════════════════════════════════════════════════════

def run_models():
    """Pipeline completo de modelado."""
    ensure_model_dirs()
    print("=" * 60)
    print("LABORATORIO 5 - Modelado")
    print("=" * 60)

    train, _, _ = load_datasets()
    X = train["text"].values
    y = train["target"].values
    groups = train["text"].str.strip().str.lower().values

    # --- Split ---
    print("\n[1/4] Division dev/holdout estratificada por grupos...")
    sp = create_group_stratified_split(X, y, groups)
    X_dev, y_dev, g_dev = sp["X_dev"], sp["y_dev"], sp["groups_dev"]
    X_hold, y_hold = sp["X_hold"], sp["y_hold"]
    print(f"  Dev: {len(X_dev)} ({len(X_dev)/len(X)*100:.1f}%)")
    print(f"  Holdout: {len(X_hold)} ({len(X_hold)/len(X)*100:.1f}%)")
    print(f"  Grupos compartidos: {sp['shared_groups']}")

    manifest = {
        "seed": SEED,
        "n_dev": int(len(X_dev)), "n_holdout": int(len(X_hold)),
        "dev_target": {str(k): int(v) for k, v
                       in pd.Series(y_dev).value_counts().items()},
        "holdout_target": {str(k): int(v) for k, v
                          in pd.Series(y_hold).value_counts().items()},
        "n_groups_dev": int(len(set(g_dev))),
        "n_groups_holdout": int(len(set(sp["groups_hold"]))),
        "shared_groups": int(sp["shared_groups"]),
    }
    with open(MODELS_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # --- CV comparison ---
    print("\n[2/4] Comparando modelos con CV estratificado por grupos...")
    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=SEED)
    configs = get_model_configs()
    results = []
    fitted = {}

    for name, (pipe, grid) in configs.items():
        print(f"  {name}...", end=" ", flush=True)

        if grid:
            search = GridSearchCV(pipe, grid, cv=inner_cv, scoring="f1",
                                  refit=True, n_jobs=-1, error_score=0.0)
            search.fit(X_dev, y_dev, groups=g_dev)
            best_pipe = search.best_estimator_
            best_params = search.best_params_
        else:
            pipe.fit(X_dev, y_dev)
            best_pipe = pipe
            best_params = {}

        y_cv = cross_val_predict(clone(best_pipe), X_dev, y_dev,
                                 groups=g_dev, cv=inner_cv)
        try:
            y_cv_p = cross_val_predict(clone(best_pipe), X_dev, y_dev,
                                       groups=g_dev, cv=inner_cv,
                                       method="predict_proba")[:, 1]
            pr_auc = average_precision_score(y_dev, y_cv_p)
            roc_auc = roc_auc_score(y_dev, y_cv_p)
        except Exception:
            pr_auc = roc_auc = np.nan

        f1 = f1_score(y_dev, y_cv)
        row = {
            "model": name, "params": str(best_params),
            "f1_disaster": round(f1, 4),
            "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
            "roc_auc": round(roc_auc, 4) if not np.isnan(roc_auc) else np.nan,
            "precision": round(precision_score(y_dev, y_cv,
                                               zero_division=0), 4),
            "recall": round(recall_score(y_dev, y_cv, zero_division=0), 4),
            "accuracy": round(accuracy_score(y_dev, y_cv), 4),
            "f1_macro": round(f1_score(y_dev, y_cv, average="macro"), 4),
        }
        results.append(row)
        fitted[name] = best_pipe
        print(f"F1={f1:.4f}")

    cv_df = pd.DataFrame(results).sort_values("f1_disaster", ascending=False)
    cv_df.to_csv(MODELS_DIR / "cv_comparison.csv", index=False)
    print("\n  Comparacion CV:")
    print(cv_df.to_string(index=False))

    best_name = cv_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\n  Modelo seleccionado: {best_name}")

    # --- Holdout ---
    print("\n[3/4] Evaluacion en holdout (unica vez)...")
    hold_result = evaluate_holdout(best_pipe, X_hold, y_hold, best_name)

    hold_preds = pd.DataFrame({
        "idx": sp["holdout_idx"], "y_true": y_hold,
        "y_pred": hold_result["y_pred"],
    })
    if "y_proba" in hold_result:
        hold_preds["y_score"] = hold_result["y_proba"]
    hold_preds.to_csv(MODELS_DIR / "holdout_predictions.csv", index=False)

    train_full = pd.read_csv(DATA_DIR / "train.csv")
    fp = (hold_result["y_pred"] == 1) & (y_hold == 0)
    fn = (hold_result["y_pred"] == 0) & (y_hold == 1)
    train_full.iloc[sp["holdout_idx"][fp]].head(15).to_csv(
        MODELS_DIR / "false_positives.csv", index=False)
    train_full.iloc[sp["holdout_idx"][fn]].head(15).to_csv(
        MODELS_DIR / "false_negatives.csv", index=False)

    import joblib
    joblib.dump(best_pipe, MODELS_DIR / "best_pipeline.joblib")

    # --- Summary ---
    print("\n[4/4] Resumen")
    print(f"  Modelo: {best_name}")
    for k, v in hold_result["metrics"].items():
        print(f"  {k}: {v}")
    print("\n" + "=" * 60)
    print("Modelado completo.")
    print("=" * 60)


def run_predict_test():
    """Reentrena con todo train.csv y predice test.csv."""
    ensure_model_dirs()
    import joblib

    print("=" * 60)
    print("PREDICCION DE test.csv")
    print("=" * 60)

    pipeline = joblib.load(MODELS_DIR / "best_pipeline.joblib")
    train, test, sample = load_datasets()

    print(f"\n[1/3] Reentrenando con todo train.csv ({len(train)} filas)...")
    pipeline.fit(train["text"].values, train["target"].values)
    joblib.dump(pipeline, MODELS_DIR / "final_pipeline.joblib")

    print(f"[2/3] Prediciendo test.csv ({len(test)} filas)...")
    y_pred = pipeline.predict(test["text"].values)

    submission = pd.DataFrame({"id": test["id"], "target": y_pred})
    submission.to_csv(DATA_DIR / "submission_lab5.csv", index=False)

    print("[3/3] Validando submission...")
    assert len(submission) == 3263, f"Filas: {len(submission)}"
    assert list(submission.columns) == ["id", "target"]
    assert (submission["id"].values == sample["id"].values).all()
    assert set(submission["target"].unique()).issubset({0, 1})

    dist = submission["target"].value_counts().sort_index()
    print(f"  Filas: {len(submission)}")
    print(f"  IDs coinciden con sample_submission: True")
    print(f"  Distribucion: 0={dist.get(0, 0)}, 1={dist.get(1, 0)}")
    print(f"  Archivo: submission_lab5.csv")

    print("\n" + "=" * 60)
    print("Prediccion completa.")
    print("=" * 60)


def validate_models():
    """Valida artefactos del modelado."""
    errors = []

    expected_files = [
        MODELS_DIR / "cv_comparison.csv",
        MODELS_DIR / "holdout_metrics.json",
        MODELS_DIR / "confusion_matrix.png",
        MODELS_DIR / "pr_roc_curves.png",
        MODELS_DIR / "holdout_predictions.csv",
        MODELS_DIR / "false_positives.csv",
        MODELS_DIR / "false_negatives.csv",
        MODELS_DIR / "split_manifest.json",
        MODELS_DIR / "best_pipeline.joblib",
    ]
    for fp in expected_files:
        if not fp.exists():
            errors.append(f"Faltante: {fp}")
        elif fp.stat().st_size < 100:
            errors.append(f"Muy pequeno: {fp}")

    # Split: zero shared groups
    manifest_path = MODELS_DIR / "split_manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        if m.get("shared_groups", -1) != 0:
            errors.append(f"Grupos compartidos != 0: {m.get('shared_groups')}")

    # Submission
    sub_path = DATA_DIR / "submission_lab5.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
        if len(sub) != 3263:
            errors.append(f"submission filas: {len(sub)}")
        if list(sub.columns) != ["id", "target"]:
            errors.append(f"submission columnas: {list(sub.columns)}")
        if not (sub["id"].values == sample["id"].values).all():
            errors.append("submission IDs no coinciden")
        if not set(sub["target"].unique()).issubset({0, 1}):
            errors.append("submission targets no binarios")
    else:
        errors.append("submission_lab5.csv no existe (ejecutar --predict-test)")

    # Proxy label check
    cv_path = MODELS_DIR / "cv_comparison.csv"
    if cv_path.exists():
        content = cv_path.read_text(encoding="utf-8")
        if "keyword_present" in content:
            errors.append("CV contiene etiqueta proxy keyword_present")

    print("=" * 60)
    print("VALIDACION MODELOS")
    print("=" * 60)
    if errors:
        print(f"\n{len(errors)} problemas:")
        for e in errors:
            print(f"  X {e}")
    else:
        print("\nTodos los controles pasaron.")
    print("=" * 60)
    return errors


# ═══════════════════════════════════════════════════════════════════════════
#  12. ANÁLISIS DE SENTIMIENTO (Fase 3)
# ═══════════════════════════════════════════════════════════════════════════

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    warnings.warn("vaderSentiment no instalado; sentimiento deshabilitado.")

from scipy.stats import mannwhitneyu
from sklearn.pipeline import FeatureUnion, make_pipeline


def _get_vader():
    if not HAS_VADER:
        raise ImportError("pip install vaderSentiment")
    return SentimentIntensityAnalyzer()


def compute_sentiment(df):
    """Aplica VADER sobre texto original (sin limpiar) preservando
    mayusculas, negaciones, puntuacion y emoticonos."""
    sia = _get_vader()
    lexicon = sia.lexicon

    records = []
    for text in df["text"]:
        scores = sia.polarity_scores(str(text))
        if scores["compound"] >= 0.05:
            label = "positiva"
        elif scores["compound"] <= -0.05:
            label = "negativa"
        else:
            label = "neutra"

        words = str(text).lower().split()
        n_pos = sum(1 for w in words if w in lexicon and lexicon[w] > 0)
        n_neg = sum(1 for w in words if w in lexicon and lexicon[w] < 0)

        records.append({
            "neg": scores["neg"],
            "neu": scores["neu"],
            "pos": scores["pos"],
            "compound": scores["compound"],
            "sentiment_label": label,
            "negativity": scores["neg"],
            "n_positive_tokens": n_pos,
            "n_negative_tokens": n_neg,
        })
    return pd.DataFrame(records, index=df.index)


def run_sentiment():
    """Pipeline completo de analisis de sentimiento."""
    ensure_sentiment_dirs()
    print("=" * 60)
    print("LABORATORIO 5 - Analisis de sentimiento")
    print("=" * 60)

    train, _, _ = load_datasets()
    sent = compute_sentiment(train)
    result = pd.concat([train, sent], axis=1)

    # --- CSV completo ---
    cols_out = ["id", "text", "target", "neg", "neu", "pos", "compound",
                "sentiment_label", "negativity",
                "n_positive_tokens", "n_negative_tokens"]
    result[cols_out].to_csv(SENTIMENT_DIR / "sentiment_scores.csv",
                            index=False)
    print(f"\n[1/6] Scores guardados: {SENTIMENT_DIR / 'sentiment_scores.csv'}")

    # --- Distribucion de etiquetas ---
    dist = result["sentiment_label"].value_counts()
    print(f"\n[2/6] Distribucion de sentimiento:")
    for lab in ["positiva", "neutra", "negativa"]:
        print(f"  {lab}: {dist.get(lab, 0)} ({dist.get(lab, 0)/len(result)*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["negativa", "neutra", "positiva"]
    colors_sent = {"negativa": "#DD8452", "neutra": "#AAAAAA", "positiva": "#4C72B0"}
    bars = [dist.get(o, 0) for o in order]
    ax.bar(order, bars, color=[colors_sent[o] for o in order])
    ax.set_ylabel("Cantidad de tweets")
    ax.set_title("Distribucion de sentimiento (VADER)")
    for i, v in enumerate(bars):
        ax.text(i, v + 30, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(SENTIMENT_DIR / "sentiment_distribution.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Comparacion por target ---
    print(f"\n[3/6] Sentimiento por target:")
    ct = pd.crosstab(result["target"].map(CLASS_LABELS),
                     result["sentiment_label"])
    ct = ct.reindex(columns=order, fill_value=0)
    print(ct.to_string())
    ct.to_csv(SENTIMENT_DIR / "sentiment_by_target.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, t in enumerate([0, 1]):
        sub = result[result["target"] == t]
        vals = [sub["sentiment_label"].value_counts().get(o, 0) for o in order]
        axes[idx].bar(order, vals, color=[colors_sent[o] for o in order])
        axes[idx].set_title(f"{CLASS_LABELS[t]} (n={len(sub)})")
        axes[idx].set_ylabel("Cantidad")
        for i, v in enumerate(vals):
            axes[idx].text(i, v + 10, str(v), ha="center", fontsize=9)
    fig.suptitle("Distribucion de sentimiento por target", fontsize=13)
    fig.tight_layout()
    fig.savefig(SENTIMENT_DIR / "sentiment_by_target.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Top 10 mas negativos y positivos ---
    print(f"\n[4/6] Tweets extremos:")
    result["_text_norm"] = result["text"].str.strip().str.lower()
    deduped = result.drop_duplicates(subset="_text_norm", keep="first")
    top_neg = deduped.nsmallest(10, "compound")[["id", "text", "target",
                                                  "compound", "sentiment_label"]]
    top_pos = deduped.nlargest(10, "compound")[["id", "text", "target",
                                                 "compound", "sentiment_label"]]
    top_neg.to_csv(SENTIMENT_DIR / "top10_negative.csv", index=False)
    top_pos.to_csv(SENTIMENT_DIR / "top10_positive.csv", index=False)
    print("  10 mas negativos:")
    for _, r in top_neg.iterrows():
        print(f"    [{CLASS_LABELS[r['target']]}] compound={r['compound']:.4f} "
              f"| {r['text'][:70]}...")
    print("  10 mas positivos:")
    for _, r in top_pos.iterrows():
        print(f"    [{CLASS_LABELS[r['target']]}] compound={r['compound']:.4f} "
              f"| {r['text'][:70]}...")

    # --- Figuras de scores por target ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(["compound", "neg", "pos"]):
        for t in [0, 1]:
            sub = result[result["target"] == t][col]
            axes[idx].hist(sub, bins=40, alpha=0.6, label=CLASS_LABELS[t],
                           color=CLASS_COLORS[t])
        axes[idx].set_title(f"Score {col} por target")
        axes[idx].set_xlabel(col)
        axes[idx].legend()
    fig.tight_layout()
    fig.savefig(SENTIMENT_DIR / "score_distributions.png",
                dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Mann-Whitney y efecto ---
    print(f"\n[5/6] Prueba Mann-Whitney (negatividad por target):")
    neg_disaster = result[result["target"] == 1]["negativity"].values
    neg_other = result[result["target"] == 0]["negativity"].values
    stat, pval = mannwhitneyu(neg_disaster, neg_other, alternative="greater")
    n1, n2 = len(neg_disaster), len(neg_other)
    # r > 0 means disaster tweets have higher negativity
    r_effect = 2 * stat / (n1 * n2) - 1
    effect_label = ("despreciable" if abs(r_effect) < 0.1
                    else "pequeno" if abs(r_effect) < 0.3
                    else "mediano" if abs(r_effect) < 0.5
                    else "grande")
    mw_results = {
        "U_statistic": float(stat),
        "p_value": float(pval),
        "n_disaster": int(n1),
        "n_other": int(n2),
        "rank_biserial_r": round(r_effect, 4),
        "effect_size_label": effect_label,
        "median_neg_disaster": round(float(np.median(neg_disaster)), 4),
        "median_neg_other": round(float(np.median(neg_other)), 4),
        "mean_neg_disaster": round(float(np.mean(neg_disaster)), 4),
        "mean_neg_other": round(float(np.mean(neg_other)), 4),
    }
    with open(SENTIMENT_DIR / "mann_whitney.json", "w", encoding="utf-8") as f:
        json.dump(mw_results, f, indent=2)
    print(f"  U = {stat:.1f}, p = {pval:.2e}")
    print(f"  r (rank-biserial) = {r_effect:.4f} ({effect_label})")
    print(f"  Mediana negatividad desastre: {mw_results['median_neg_disaster']}")
    print(f"  Mediana negatividad no-desastre: {mw_results['median_neg_other']}")
    if pval < 0.05:
        print("  Conclusion: diferencia estadisticamente significativa (p < 0.05),")
        print(f"  pero el efecto es {effect_label} — la utilidad practica es limitada"
              " si r < 0.3.")
    else:
        print("  Conclusion: no se encontro diferencia significativa (p >= 0.05).")

    # --- Explicacion emoticonos ---
    print(f"\n[6/6] Nota sobre preservacion de emoticonos y puntuacion:")
    print("  VADER fue disenado para texto informal y redes sociales.")
    print("  Capitalizar (ej. 'HORRIBLE') amplifica la valencia.")
    print("  Puntuacion (ej. '!!!') refuerza la intensidad.")
    print("  Emoticonos (ej. ':)', ':(') tienen scores propios en el lexico.")
    print("  Negaciones (ej. 'not good') invierten la polaridad del siguiente token.")
    print("  Eliminar estos elementos degradaria la calidad del analisis.")

    print("\n" + "=" * 60)
    print("Analisis de sentimiento completo.")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════
#  13. MODELO CON NEGATIVIDAD (Fase 3 - ablacion)
# ═══════════════════════════════════════════════════════════════════════════

class NegativityTransformer(BaseEstimator, TransformerMixin):
    """Calcula negativity (VADER neg score) desde texto crudo.
    Compatible con Pipeline: fit/transform devuelve array (n, 1)."""

    def __init__(self):
        self._sia = None

    def fit(self, X, y=None):
        self._sia = _get_vader()
        return self

    def transform(self, X):
        sia = self._sia or _get_vader()
        return np.array(
            [sia.polarity_scores(str(t))["neg"] for t in X]
        ).reshape(-1, 1)


from scipy.sparse import hstack as sp_hstack, issparse


class TfidfPlusNegativity(BaseEstimator, TransformerMixin):
    """Combina TF-IDF de caracteres + negativity en una sola etapa.
    Acepta texto crudo, aplica TextCleaner internamente para TF-IDF
    y NegativityTransformer sobre texto original."""

    def __init__(self, tfidf_params=None):
        self.tfidf_params = tfidf_params

    def fit(self, X, y=None):
        params = {} if self.tfidf_params is None else dict(self.tfidf_params)
        self._cleaner = TextCleaner()
        self._tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_df=0.95, sublinear_tf=True,
            **params,
        )
        self._neg = NegativityTransformer()
        cleaned = self._cleaner.fit_transform(X)
        self._tfidf.fit(cleaned)
        self._neg.fit(X)
        return self

    def transform(self, X):
        cleaned = self._cleaner.transform(X)
        tfidf_mat = self._tfidf.transform(cleaned)
        neg_mat = self._neg.transform(X)
        return sp_hstack([tfidf_mat, neg_mat])


def run_models_sentiment():
    """Ablacion: LogReg_char baseline vs LogReg_char + negativity."""
    ensure_sentiment_dirs()
    ensure_model_dirs()
    print("=" * 60)
    print("LABORATORIO 5 - Ablacion: negatividad")
    print("=" * 60)

    train, _, _ = load_datasets()
    X = train["text"].values
    y = train["target"].values
    groups = train["text"].str.strip().str.lower().values

    sp = create_group_stratified_split(X, y, groups)
    X_dev, y_dev, g_dev = sp["X_dev"], sp["y_dev"], sp["groups_dev"]
    X_hold, y_hold = sp["X_hold"], sp["y_hold"]

    inner_cv = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                    random_state=SEED)
    best_C = 0.5

    # --- Baseline: LogReg_char exacto ---
    print("\n[1/3] Baseline LogReg_char (C=0.5)...")
    pipe_base = Pipeline([
        ("cleaner", TextCleaner()),
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            min_df=2, max_df=0.95, sublinear_tf=True,
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED,
                                   class_weight="balanced", C=best_C)),
    ])
    y_cv_base = cross_val_predict(clone(pipe_base), X_dev, y_dev,
                                  groups=g_dev, cv=inner_cv)
    y_cv_p_base = cross_val_predict(clone(pipe_base), X_dev, y_dev,
                                    groups=g_dev, cv=inner_cv,
                                    method="predict_proba")[:, 1]
    f1_cv_base = round(f1_score(y_dev, y_cv_base), 4)
    prauc_cv_base = round(average_precision_score(y_dev, y_cv_p_base), 4)
    print(f"  CV F1={f1_cv_base}, PR-AUC={prauc_cv_base}")

    pipe_base.fit(X_dev, y_dev)
    yp_base = pipe_base.predict(X_hold)
    ypr_base = pipe_base.predict_proba(X_hold)[:, 1]
    f1_hold_base = round(f1_score(y_hold, yp_base), 4)
    prauc_hold_base = round(average_precision_score(y_hold, ypr_base), 4)
    print(f"  Holdout F1={f1_hold_base}, PR-AUC={prauc_hold_base}")

    # --- LogReg_char + negativity ---
    print("\n[2/3] LogReg_char + negativity (C=0.5)...")
    pipe_neg = Pipeline([
        ("features", TfidfPlusNegativity()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED,
                                   class_weight="balanced", C=best_C)),
    ])
    y_cv_neg = cross_val_predict(clone(pipe_neg), X_dev, y_dev,
                                 groups=g_dev, cv=inner_cv)
    y_cv_p_neg = cross_val_predict(clone(pipe_neg), X_dev, y_dev,
                                   groups=g_dev, cv=inner_cv,
                                   method="predict_proba")[:, 1]
    f1_cv_neg = round(f1_score(y_dev, y_cv_neg), 4)
    prauc_cv_neg = round(average_precision_score(y_dev, y_cv_p_neg), 4)
    print(f"  CV F1={f1_cv_neg}, PR-AUC={prauc_cv_neg}")

    pipe_neg.fit(X_dev, y_dev)
    yp_neg = pipe_neg.predict(X_hold)
    ypr_neg = pipe_neg.predict_proba(X_hold)[:, 1]
    f1_hold_neg = round(f1_score(y_hold, yp_neg), 4)
    prauc_hold_neg = round(average_precision_score(y_hold, ypr_neg), 4)
    print(f"  Holdout F1={f1_hold_neg}, PR-AUC={prauc_hold_neg}")

    # --- Comparacion ---
    print("\n[3/3] Comparacion ablacion:")
    comparison = {
        "baseline": {
            "model": "LogReg_char",
            "C": best_C,
            "cv_f1": f1_cv_base,
            "cv_pr_auc": prauc_cv_base,
            "holdout_f1": f1_hold_base,
            "holdout_pr_auc": prauc_hold_base,
        },
        "baseline_plus_negativity": {
            "model": "LogReg_char + negativity",
            "C": best_C,
            "cv_f1": f1_cv_neg,
            "cv_pr_auc": prauc_cv_neg,
            "holdout_f1": f1_hold_neg,
            "holdout_pr_auc": prauc_hold_neg,
        },
        "delta": {
            "cv_f1": round(f1_cv_neg - f1_cv_base, 4),
            "cv_pr_auc": round(prauc_cv_neg - prauc_cv_base, 4),
            "holdout_f1": round(f1_hold_neg - f1_hold_base, 4),
            "holdout_pr_auc": round(prauc_hold_neg - prauc_hold_base, 4),
        },
    }
    d_cv = comparison["delta"]["cv_f1"]
    d_hold = comparison["delta"]["holdout_f1"]
    if d_cv < 0 and d_hold < 0:
        conclusion = (
            f"Agregar negatividad no mejoro el clasificador: "
            f"F1 disminuyo {abs(d_cv):.4f} en CV y {abs(d_hold):.4f} en holdout. "
            f"El deterioro es pequeno, pero consistente; "
            f"se conserva LogReg_char sin negatividad."
        )
    elif d_cv > 0 and d_hold > 0:
        conclusion = (
            f"Agregar negatividad mejoro el clasificador: "
            f"F1 aumento {d_cv:.4f} en CV y {d_hold:.4f} en holdout."
        )
    else:
        conclusion = (
            f"Resultado mixto: delta CV F1 = {d_cv:+.4f}, "
            f"delta holdout F1 = {d_hold:+.4f}. "
            f"Sin evidencia consistente de mejora; "
            f"se conserva LogReg_char sin negatividad."
        )
    comparison["conclusion"] = conclusion

    with open(SENTIMENT_DIR / "ablation_results.json", "w",
              encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    comp_df = pd.DataFrame([
        {"modelo": "LogReg_char", "cv_f1": f1_cv_base,
         "cv_pr_auc": prauc_cv_base, "holdout_f1": f1_hold_base,
         "holdout_pr_auc": prauc_hold_base},
        {"modelo": "LogReg_char + negativity", "cv_f1": f1_cv_neg,
         "cv_pr_auc": prauc_cv_neg, "holdout_f1": f1_hold_neg,
         "holdout_pr_auc": prauc_hold_neg},
    ])
    comp_df.to_csv(SENTIMENT_DIR / "ablation_comparison.csv", index=False)

    print(f"  {'Metrica':<20} {'Baseline':>10} {'+ Negat.':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    print(f"  {'CV F1':<20} {f1_cv_base:>10.4f} {f1_cv_neg:>10.4f} "
          f"{comparison['delta']['cv_f1']:>+10.4f}")
    print(f"  {'CV PR-AUC':<20} {prauc_cv_base:>10.4f} {prauc_cv_neg:>10.4f} "
          f"{comparison['delta']['cv_pr_auc']:>+10.4f}")
    print(f"  {'Holdout F1':<20} {f1_hold_base:>10.4f} {f1_hold_neg:>10.4f} "
          f"{comparison['delta']['holdout_f1']:>+10.4f}")
    print(f"  {'Holdout PR-AUC':<20} {prauc_hold_base:>10.4f} {prauc_hold_neg:>10.4f} "
          f"{comparison['delta']['holdout_pr_auc']:>+10.4f}")
    print(f"\n  Conclusion: {conclusion}")

    print("\n" + "=" * 60)
    print("Ablacion completa.")
    print("=" * 60)


def validate_sentiment():
    """Valida artefactos de sentimiento."""
    errors = []
    expected = [
        SENTIMENT_DIR / "sentiment_scores.csv",
        SENTIMENT_DIR / "sentiment_distribution.png",
        SENTIMENT_DIR / "sentiment_by_target.png",
        SENTIMENT_DIR / "sentiment_by_target.csv",
        SENTIMENT_DIR / "score_distributions.png",
        SENTIMENT_DIR / "top10_negative.csv",
        SENTIMENT_DIR / "top10_positive.csv",
        SENTIMENT_DIR / "mann_whitney.json",
        SENTIMENT_DIR / "ablation_results.json",
        SENTIMENT_DIR / "ablation_comparison.csv",
    ]
    for fp in expected:
        if not fp.exists():
            errors.append(f"Faltante: {fp}")
        elif fp.stat().st_size < 50:
            errors.append(f"Muy pequeno: {fp}")

    if (SENTIMENT_DIR / "sentiment_scores.csv").exists():
        sc = pd.read_csv(SENTIMENT_DIR / "sentiment_scores.csv")
        for col in ["neg", "neu", "pos", "compound", "sentiment_label",
                     "negativity", "n_positive_tokens", "n_negative_tokens"]:
            if col not in sc.columns:
                errors.append(f"Columna faltante en scores: {col}")
        if len(sc) != 7613:
            errors.append(f"Filas scores: {len(sc)}, esperado 7613")

    if (SENTIMENT_DIR / "mann_whitney.json").exists():
        mw = json.loads((SENTIMENT_DIR / "mann_whitney.json").read_text(
            encoding="utf-8"))
        if "p_value" not in mw:
            errors.append("mann_whitney.json sin p_value")
        if "rank_biserial_r" not in mw:
            errors.append("mann_whitney.json sin rank_biserial_r")

    print("=" * 60)
    print("VALIDACION SENTIMIENTO")
    print("=" * 60)
    if errors:
        print(f"\n{len(errors)} problemas:")
        for e in errors:
            print(f"  X {e}")
    else:
        print("\nTodos los controles pasaron.")
    print("=" * 60)
    return errors


def validate_final():
    """Validacion integral del proyecto."""
    errors = []

    print("=" * 60)
    print("VALIDACION FINAL - Laboratorio 5")
    print("=" * 60)

    # 1. Sub-validaciones
    print("\n[1/9] Validaciones previas...")
    for name, fn in [("EDA", validate_eda), ("Modelos", validate_models),
                     ("Sentimiento", validate_sentiment)]:
        sub = fn()
        if sub:
            for e in sub:
                errors.append(f"[{name}] {e}")

    # 2. Notebook sin errores
    print("\n[2/9] Notebook...")
    nb_path = Path("lab5.ipynb")
    if nb_path.exists():
        import nbformat
        nb = nbformat.read(str(nb_path), as_version=4)
        for i, c in enumerate(nb.cells):
            if c.cell_type == "code" and c.outputs:
                for out in c.outputs:
                    if out.get("output_type") == "error":
                        errors.append(
                            f"Notebook celda {i}: error {out.get('ename', '?')}")
    else:
        errors.append("lab5.ipynb no encontrado")

    # 3. Informe y tutorial sin PENDIENTE
    print("\n[3/9] Documentos...")
    for doc in ["INFORME_LAB5.md", "GUIA_REPLICACION_LAB5.md"]:
        p = Path(doc)
        if p.exists():
            content = p.read_text(encoding="utf-8")
            if "PENDIENTE" in content.upper():
                errors.append(f"{doc} contiene PENDIENTE")
        else:
            errors.append(f"{doc} no encontrado")

    # 4. Nombres y repositorio
    print("\n[4/9] Metadatos...")
    informe = Path("INFORME_LAB5.md")
    if informe.exists():
        txt = informe.read_text(encoding="utf-8")
        for name in ["Mendez Alvarado", "Yee Vidal"]:
            if name not in txt:
                errors.append(f"Nombre '{name}' no encontrado en informe")
        if "Paul-1511/lab5data" not in txt:
            errors.append("URL del repositorio no encontrada en informe")

    # 5. Conclusion de ablacion correcta
    print("\n[5/9] Conclusion de ablacion...")
    abl_path = SENTIMENT_DIR / "ablation_results.json"
    if abl_path.exists():
        abl = json.loads(abl_path.read_text(encoding="utf-8"))
        conclusion = abl.get("conclusion", "")
        if "|delta F1| < 0.005" in conclusion:
            errors.append("Conclusion de ablacion contiene frase incorrecta "
                          "'|delta F1| < 0.005'")
        if "no mejoro" not in conclusion and "no mejoro" not in conclusion.lower():
            delta_cv = abl.get("delta", {}).get("cv_f1", 0)
            delta_hold = abl.get("delta", {}).get("holdout_f1", 0)
            if delta_cv < 0 and delta_hold < 0:
                errors.append("Conclusion de ablacion no refleja deterioro "
                              "consistente")
    else:
        errors.append("ablation_results.json no encontrado")

    # 6. Figuras referenciadas en informe
    print("\n[6/9] Figuras referenciadas...")
    if informe.exists():
        txt = informe.read_text(encoding="utf-8")
        import re as _re
        imgs = _re.findall(r"!\[.*?\]\((.+?)\)", txt)
        for img in imgs:
            if not Path(img).exists():
                errors.append(f"Figura referenciada no existe: {img}")

    # 7. Submission valida
    print("\n[7/9] Submission...")
    sub_path = DATA_DIR / "submission_lab5.csv"
    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        if len(sub) != 3263:
            errors.append(f"submission filas: {len(sub)}")
        if not set(sub["target"].unique()).issubset({0, 1}):
            errors.append("submission targets no binarios")
    else:
        errors.append("submission_lab5.csv no encontrado")

    # 8. PDF
    print("\n[8/9] PDF...")
    pdf_path = Path("INFORME_LAB5.pdf")
    if pdf_path.exists():
        size = pdf_path.stat().st_size
        if size < 1000:
            errors.append(f"PDF muy pequeno: {size} bytes")
        header = pdf_path.read_bytes()[:5]
        if header != b"%PDF-":
            errors.append("PDF no tiene header valido")
    else:
        errors.append("INFORME_LAB5.pdf no encontrado (ejecutar md2pdf.py)")

    # 9. Ausencia de rutas absolutas y etiqueta proxy
    print("\n[9/9] Controles de higiene...")
    _abs_markers = ["C:" + "\\", "/home" + "/", "/Users" + "/"]
    script_lines = Path("analisis_lab5.py").read_text(encoding="utf-8").splitlines()
    for ln_no, ln in enumerate(script_lines, 1):
        if "_abs_markers" in ln:
            continue
        if any(m in ln for m in _abs_markers):
            errors.append(f"analisis_lab5.py:{ln_no} contiene ruta absoluta")
    cv_path = MODELS_DIR / "cv_comparison.csv"
    if cv_path.exists():
        cv_txt = cv_path.read_text(encoding="utf-8")
        if "keyword_present" in cv_txt:
            errors.append("CV contiene etiqueta proxy keyword_present")

    # Resumen
    print("\n" + "=" * 60)
    if errors:
        print(f"{len(errors)} problemas encontrados:")
        for e in errors:
            print(f"  X {e}")
    else:
        print("TODOS LOS CONTROLES PASARON.")
    print("=" * 60)
    return errors


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Lab 5 - Clasificacion de tweets sobre desastres",
    )
    parser.add_argument("--eda", action="store_true",
                        help="Pipeline EDA completo")
    parser.add_argument("--validate-eda", action="store_true",
                        help="Valida artefactos EDA")
    parser.add_argument("--models", action="store_true",
                        help="Entrena y compara modelos")
    parser.add_argument("--predict-test", action="store_true",
                        help="Reentrena y predice test.csv")
    parser.add_argument("--validate-models", action="store_true",
                        help="Valida artefactos de modelado")
    parser.add_argument("--sentiment", action="store_true",
                        help="Analisis de sentimiento VADER")
    parser.add_argument("--models-sentiment", action="store_true",
                        help="Ablacion: LogReg_char +/- negatividad")
    parser.add_argument("--validate-sentiment", action="store_true",
                        help="Valida artefactos de sentimiento")
    parser.add_argument("--validate-final", action="store_true",
                        help="Validacion integral del proyecto")
    args = parser.parse_args()

    if args.eda:
        run_eda()
    elif args.validate_eda:
        validate_eda()
    elif args.models:
        run_models()
    elif args.predict_test:
        run_predict_test()
    elif args.validate_models:
        validate_models()
    elif args.sentiment:
        run_sentiment()
    elif args.models_sentiment:
        run_models_sentiment()
    elif args.validate_sentiment:
        validate_sentiment()
    elif args.validate_final:
        validate_final()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
