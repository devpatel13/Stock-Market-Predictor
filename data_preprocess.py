#!/usr/bin/env python3
"""
Phase 1 Data Ingestion Pipeline
================================
Production-grade, configuration-driven ingestion of financial datasets.

Applies folder-specific parsing rules (FOLDER_CONFIGS), standardizes
dates/text/sentiment, deduplicates, and saves .parquet outputs to
processed_phase1/ while preserving the original folder hierarchy.

Logging : ALL detail  ->  phase1_ingestion.log  (file ONLY - stdout stays clean)
Terminal: ONLY the final high-level summary report
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import logging
import os
import re
import shutil
import time
import traceback
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser

# =============================================================================
# 1. THE RULEBOOK - folder-level parsing configuration
# =============================================================================
# Keys MUST exactly match the immediate parent folder name inside datasets/.
# Values are plain dicts; all keys are optional - unspecified folders fall
# through to the global defaults defined further below.

FOLDER_CONFIGS: Dict[str, Dict[str, Any]] = {
    # -- Sentiment Analysis for Financial News --------------------------------
    # .txt files: '@'-delimited, column order text then sentiment.
    # all-data.csv: no header row, column order sentiment then text.
    "Sentiment Analysis for Financial News": {
        "txt_sep": "@",
        "txt_names": ["text", "sentiment"],
        # set of filenames (case-insensitive match) that carry no header row
        "headerless_csv": {"all-data.csv"},
        "headerless_names": ["sentiment", "text"],
    },

    # -- Stock market prediction business news India --------------------------
    # Wide-format Excel: multiple headline columns -> melt into single 'text'.
    "Stock market prediction business news India": {
        "excel_melt": True,
    },

    # -- Stock Market Prediction  India news headlines ------------------------
    # Date column is a compact integer string with no delimiters: YYYYMMDD.
    "Stock Market Prediction  India news headlines": {
        "date_format": "%Y%m%d",
    },

    # -- Indian financial news articles 2003 to 2020 --------------------------
    # Complex date strings e.g. "May 26, 2020, Tuesday" -> dateutil.parser.
    "Indian financial news articles 2003 to 2020": {
        "complex_dates": True,
    },

    # -- News_sentiments_india 2001 to 2022 -----------------------------------
    # Sentiment column stores floats: -1.0 / 0.0 / 1.0 -> canonical strings.
    "News_sentiments_india 2001 to 2022": {
        "sentiment_float_map": {-1.0: "Negative", 0.0: "Neutral", 1.0: "Positive"},
    },

    # -- Business News Headlines with Sentiment (2017-2021) -------------------
    # Trailing commas create spurious 'Unnamed: N' columns -> drop them all.
    "Business News Headlines with Sentiment (2017-2021)": {
        "drop_unnamed": True,
        "dayfirst": True,  # ADD THIS
    },

    # -- DD-MM-YYYY date formats ----------------------------------------------
    "Business News Description with sentiment": {
        "dayfirst": True,
        "drop_unnamed": True,
    },
    "Economic Times Headlines India 2022 to 2025": {
        "dayfirst": True,
    },
}

# File-specific loading overrides (filename -> loader options).
# Keys are lower-cased filenames for robust matching independent of folder path.
FILE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Sentiment Analysis for Financial News
    "all-data.csv": {
        "header": None,
        "names": ["sentiment", "text"],
        "encoding": "latin1",
    },
    "sentences_50agree.txt": {
        "sep": "@",
        "header": None,
        "names": ["text", "sentiment"],
    },
    "sentences_66agree.txt": {
        "sep": "@",
        "header": None,
        "names": ["text", "sentiment"],
    },
    "sentences_75agree.txt": {
        "sep": "@",
        "header": None,
        "names": ["text", "sentiment"],
    },
    "sentences_allagree.txt": {
        "sep": "@",
        "header": None,
        "names": ["text", "sentiment"],
    },

    # Stock market prediction business news India (wide-format Excel -> melt)
    "mnews_2014.xlsx": {"excel_melt": True},
    "mnews_2015.xlsx": {"excel_melt": True},
    "mnews_2016.xlsx": {"excel_melt": True},
    "mnews_2017.xlsx": {"excel_melt": True},
    "mnews_2018.xlsx": {"excel_melt": True},
    "mnews_2019.xlsx": {"excel_melt": True},
    "mnews_2020.xlsx": {"excel_melt": True},

    # Headerless news rows: first row is data, not column names
    "india-news-headlines.csv": {
        "header": None,
        "names": ["Date", "Category", "Headline"],
    },

    # Schema alignments / cleanup
    "final_news_sentiment_analysis.csv": {
        "header": None,
        "names": ["company", "symbol", "text", "date", "sentiment"],
    },
    
    "gnew_list_2021_processed.csv": {
        "rename_columns": {"Text": "text", "Date": "date"},
        "drop_columns": ["Unnamed: 2"], # Keep this just in case 2021 has trailing commas
    },
    "gnew_list_2022_processed.csv": {
        "rename_columns": {"Text": "text", "Date": "date"},
    },
    "gnew_list_2023_processed.csv": {
        "rename_columns": {"Text": "text", "Date": "date"},
    },

    # No timestamp datasets: bypass temporal alignment stage
    "training_data_26000.csv": {
        "skip_temporal_alignment": True,
    },
    "data_raw.csv": {
        "skip_temporal_alignment": True,
    },
    "info.csv": {
        "skip_temporal_alignment": True,
    },
}

# =============================================================================
# 2. CONFIGURATION & RESULT DATACLASSES
# =============================================================================

@dataclass
class PreprocessConfig:
    input_root: str = "datasets"
    output_root: str = "processed_phase1"
    workers: int = 24
    log_file: str = "phase1_ingestion.log"


@dataclass
class ProcessResult:
    status: str                        # "success" | "failed"
    input_file: str
    output_file: Optional[str]
    rows_in: int
    rows_out: int
    folder_config_key: Optional[str]   # which FOLDER_CONFIGS entry fired
    date_col: Optional[str]
    text_cols: List[str]
    error: Optional[str] = None
    # Worker accumulates structured log messages; main process writes them.
    log_messages: List[str] = field(default_factory=list)


# =============================================================================
# 3. COLUMN IDENTIFICATION
# =============================================================================

class ColumnMapper:
    # NOTE: no \b word boundaries — underscore is a \w char in Python regex,
    # so \b would fail on compound names like 'publish_date' or 'headline_text'.
    _DATE_PAT   = re.compile(
        r"(date|time|timestamp|datetime|publish|created|updated|posted|dt)", re.I
    )
    _TEXT_PAT   = re.compile(
        r"(headline|title|description|text|content)", re.I
    )
    _TARGET_PAT = re.compile(r"(sentiment|label)", re.I)

    @classmethod
    def identify_date_column(cls, df: pd.DataFrame) -> Optional[str]:
        if df.empty:
            return None
        scores: List[Tuple[int, str]] = []
        for col in df.columns:
            name = str(col).strip()
            
            # --- PATCH 1: Stop 'senTIMEnt' from being recognized as a date ---
            if cls._TARGET_PAT.search(name):
                continue
            # -----------------------------------------------------------------
            
            score = 0
            if cls._DATE_PAT.search(name):
                score += 5
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                score += 4
            if (pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])):
                sample = df[col].dropna().astype(str).head(200)
                if not sample.empty:
                    try:
                        dt_ratio = (
                            pd.to_datetime(
                                sample, format="mixed", dayfirst=True,
                                errors="coerce", utc=True,
                            ).notna().mean()
                        )
                    except Exception:
                        dt_ratio = pd.to_datetime(
                            sample, errors="coerce", utc=True
                        ).notna().mean()
                    if dt_ratio >= 0.6:
                        score += 3
                    elif dt_ratio >= 0.3:
                        score += 1
            if score > 0:
                scores.append((score, name))
        if not scores:
            return None
        scores.sort(key=lambda x: (-x[0], x[1]))
        return scores[0][1]

    @classmethod
    def identify_text_columns(cls, df: pd.DataFrame) -> List[str]:
        return [
            str(c) for c in df.columns
            if not cls._TARGET_PAT.search(str(c).strip())
            and cls._TEXT_PAT.search(str(c).strip())
            and (
                pd.api.types.is_object_dtype(df[c])
                or pd.api.types.is_string_dtype(df[c])
            )
        ]

    @classmethod
    def identify_target_columns(cls, df: pd.DataFrame) -> List[str]:
        return [
            str(c) for c in df.columns
            if cls._TARGET_PAT.search(str(c).strip())
        ]


# =============================================================================
# 4. TEXT NORMALIZER
# =============================================================================

class TextNormalizer:
    _HTML_TAG   = re.compile(r"<[^>]+>")
    _MULTISPACE = re.compile(r"\s+")
    # Mojibake sequences produced by reading UTF-8 bytes as Latin-1/Windows-1252.
    # All non-ASCII characters expressed as \uXXXX escapes to avoid encoding issues.
    _ARTIFACT = re.compile(
        r"(?:"
        r"\u00e2\u20ac\u2122"   # RIGHT SINGLE QUOTATION MARK mojibake
        r"|\u00e2\u20ac\u0153"  # LEFT DOUBLE QUOTATION MARK mojibake
        r"|\u00e2\u20ac\x9d"    # RIGHT DOUBLE QUOTATION MARK mojibake
        r"|\u00e2\u20ac\u2014"  # EM DASH mojibake
        r"|\u00e2\u20ac\u2013"  # EN DASH mojibake
        r"|\u00e2\u20ac\u2026"  # ELLIPSIS mojibake
        r"|\u00c2"              # non-breaking space / Latin-1 mojibake leader
        r"|\u00c3"              # Latin extended mojibake leader
        r")",
        re.I,
    )

    _CONTRACTIONS: Dict[str, str] = {
        "can't":   "cannot",    "won't":    "will not",  "n't":    " not",
        "'re":     " are",      "'ve":      " have",     "'ll":    " will",
        "'d":      " would",    "'m":       " am",       "it's":   "it is",
        "that's":  "that is",   "what's":   "what is",   "there's":"there is",
        "here's":  "here is",   "isn't":    "is not",    "aren't": "are not",
        "wasn't":  "was not",   "weren't":  "were not",  "don't":  "do not",
        "doesn't": "does not",  "didn't":   "did not",   "hasn't": "has not",
        "haven't": "have not",  "hadn't":   "had not",
    }
    _CONTRACTION_RE = re.compile(
        r"\b("
        + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True))
        + r")\b",
        re.I,
    )

    @classmethod
    def _fix_unicode(cls, text: str) -> str:
        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        return cls._ARTIFACT.sub(" ", text)

    @classmethod
    def _expand_contractions(cls, text: str) -> str:
        return cls._CONTRACTION_RE.sub(
            lambda m: cls._CONTRACTIONS.get(m.group(0).lower(), m.group(0)), text
        )

    @classmethod
    def clean_series(cls, s: pd.Series) -> pd.Series:
        """
        Normalize a text column: strip HTML, fix unicode, expand contractions,
        remove non-ASCII, lowercase, collapse whitespace.

        ONLY call on columns whose name matches headline/title/text/content/
        description.  Never call on sentiment/label columns (guarded upstream
        by ColumnMapper.identify_text_columns which excludes TARGET_PAT names).
        """
        s = s.copy()
        non_null_mask = s.notna()
        if non_null_mask.any():
            s.loc[non_null_mask] = s.loc[non_null_mask].astype(str)
        s = s.fillna("")
        s = s.str.replace(cls._HTML_TAG, " ", regex=True)
        s = s.map(cls._fix_unicode)
        s = s.map(cls._expand_contractions)
        s = s.str.encode("ascii", errors="ignore").str.decode("ascii")
        s = s.str.lower()
        s = s.str.replace(cls._MULTISPACE, " ", regex=True).str.strip()
        return s


# =============================================================================
# 5. ROBUST FILE LOADER  (utf-8 -> latin1 -> utf-8/replace)
# =============================================================================

def _load_file(
    path: Path,
    *,
    sep: str = ",",
    header: Any = "infer",
    names: Optional[List[str]] = None,
    index_col: Any = None,
    encoding: Optional[str] = None,
) -> pd.DataFrame:
    """
    CSV/TXT reader with a three-stage encoding fallback strategy:
      1. utf-8   (strict)
      2. latin1  (strict)  - covers Windows-1252 / ISO-8859-1 sourced data
      3. utf-8 with 'replace' error handler - last-resort, prevents crash

    Silently skips malformed lines (on_bad_lines='skip').
    """
    read_kwargs: Dict[str, Any] = dict(
        sep=sep,
        header=header,
        names=names,
        index_col=index_col,
        low_memory=False,
        on_bad_lines="skip",
    )
    attempts: List[Tuple[str, str]] = []
    if encoding:
        attempts.append((encoding, "strict"))
    for enc_try in ["utf-8", "latin1"]:
        if enc_try != encoding:
            attempts.append((enc_try, "strict"))
    attempts.append(("utf-8", "replace"))

    for enc, err_mode in attempts:
        try:
            return pd.read_csv(
                path, encoding=enc, encoding_errors=err_mode, **read_kwargs
            )
        except UnicodeDecodeError:
            continue
    # Should never reach here after the 'replace' fallback, but guard anyway.
    raise RuntimeError(f"Could not decode {path} with any supported encoding")


# =============================================================================
# 6. FOLDER-SPECIFIC DISPATCH  (driven entirely by FOLDER_CONFIGS)
# =============================================================================

def _apply_file_config(
    path: Path,
    log: List[str],
    folder_cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Apply loading rules based on exact filename matches, respecting both
    FILE_CONFIGS (file-level) and FOLDER_CONFIGS (folder-level) directives.
    Returns a raw DataFrame; all downstream transforms happen in _process_file.
    """
    if folder_cfg is None:
        folder_cfg = {}

    ext = path.suffix.lower()
    file_cfg = FILE_CONFIGS.get(path.name.lower())
    file_name_lower = path.name.lower()

    if not file_cfg:
        return _default_load(path, log)

    if file_cfg.get("excel_melt") and ext in (".xlsx", ".xls"):
        log.append("  rule: Excel wide-format -> read_excel() + melt()")
        return _read_and_melt_excel(path, log)

    sep = file_cfg.get("sep")
    header = file_cfg.get("header")
    names = file_cfg.get("names")
    index_col = file_cfg.get("index_col")
    encoding = file_cfg.get("encoding")

    # Check folder-level config for headerless CSV or .txt files
    if header is None and names is None:
        if ext == ".csv" and folder_cfg.get("headerless_csv"):
            if file_name_lower in folder_cfg["headerless_csv"]:
                header = None
                names = folder_cfg.get("headerless_names")
                log.append(
                    f"  rule: FILE_CONFIGS + FOLDER_CONFIGS (headerless_csv)"
                )
        elif ext == ".txt" and folder_cfg.get("txt_sep"):
            if sep is None:
                sep = folder_cfg["txt_sep"]
            if names is None:
                names = folder_cfg.get("txt_names")
            log.append(
                f"  rule: FILE_CONFIGS + FOLDER_CONFIGS (.txt parsing)"
            )
        else:
            log.append(
                f"  rule: FILE_CONFIGS match for '{file_name_lower}'"
            )
    else:
        log.append(
            f"  rule: FILE_CONFIGS match for '{file_name_lower}'"
        )

    # Apply defaults if not set by file or folder config
    if sep is None:
        sep = ","
    if header is None and "header" not in file_cfg:
        header = "infer"

    # Route based on file extension
    if ext in (".csv", ".txt"):
        return _load_file(
            path,
            sep=sep,
            header=header,
            names=names,
            index_col=index_col,
            encoding=encoding,
        )
    elif ext in (".xlsx", ".xls"):
        log.append("  rule: FILE_CONFIGS with Excel file -> read_excel()")
        engine = "openpyxl" if ext == ".xlsx" else None
        return pd.read_excel(path, engine=engine)
    elif ext == ".parquet":
        log.append("  rule: FILE_CONFIGS with Parquet file -> read_parquet()")
        return pd.read_parquet(path, engine="pyarrow")
    elif ext in (".json", ".jsonl"):
        log.append("  rule: FILE_CONFIGS with JSON file -> read_json()")
        try:
            return pd.read_json(path, lines=(ext == ".jsonl"))
        except ValueError:
            return pd.read_json(path, lines=True)
    else:
        # Fallback: use generic loader
        return _load_file(
            path,
            sep=sep,
            header=header,
            names=names,
            index_col=index_col,
            encoding=encoding,
        )


def _read_and_melt_excel(path: Path, log: List[str]) -> pd.DataFrame:
    """
    Read a wide-format Excel file, then melt all non-date columns into a
    single 'text' column.  Each row in the output represents one headline
    paired with its original date.
    """
    engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
    df = pd.read_excel(path, engine=engine)

    date_col  = ColumnMapper.identify_date_column(df)
    id_vars   = [date_col] if date_col else []
    val_vars  = [c for c in df.columns if c != date_col]

    melted = (
        df.melt(
            id_vars=id_vars, value_vars=val_vars,
            var_name="_src_col", value_name="text",
        )
        .drop(columns=["_src_col"])
        .dropna(subset=["text"])
    )
    # Normalise date column name for consistency
    if date_col and date_col != "date":
        melted = melted.rename(columns={date_col: "date"})

    log.append(
        f"  melt: {len(df)} rows x {len(df.columns)} cols"
        f" -> {len(melted)} rows (id_vars={id_vars})"
    )
    return melted.reset_index(drop=True)


def _default_load(path: Path, log: List[str]) -> pd.DataFrame:
    """Universal loader for files that have no special folder overrides."""
    ext = path.suffix.lower()

    if ext in (".csv", ".txt"):
        return _load_file(path)

    if ext == ".tsv":
        return _load_file(path, sep="\t")

    if ext in (".xlsx", ".xls"):
        log.append("  rule: default Excel read")
        engine = "openpyxl" if ext == ".xlsx" else None
        return pd.read_excel(path, engine=engine)

    if ext == ".parquet":
        return pd.read_parquet(path, engine="pyarrow")

    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path, lines=(ext == ".jsonl"))
        except ValueError:
            return pd.read_json(path, lines=True)

    if ext == "":
        # Extensionless file - sniff delimiter
        log.append("  rule: extensionless file - delimiter sniff")
        best_df = None
        
        # --- PATCH 2: Try comma FIRST, and verify it actually split the columns ---
        for sep in (",", "\t", ";", "|"):
            try:
                df = _load_file(path, sep=sep)
                if best_df is None:
                    best_df = df
                
                # If it successfully split into multiple columns, we found the right separator!
                if len(df.columns) > 1:
                    return df
            except Exception:
                continue
        
        if best_df is not None:
            return best_df
        # --------------------------------------------------------------------------

    raise ValueError(f"Unsupported file extension: {path.suffix!r}")


# =============================================================================
# 7. DATE STANDARDISATION  ->  ISO 8601  (YYYY-MM-DD HH:MM:SS, UTC)
# =============================================================================

def _safe_dateutil_parse(value: Any) -> Any:
    """Parse a single value with dateutil; returns NaT on any failure."""
    if pd.isna(value):
        return pd.NaT
    try:
        return date_parser.parse(str(value), dayfirst=True)
    except Exception:
        return pd.NaT


def _parse_datetime_series(
    series: pd.Series,
    *,
    date_format: Optional[str] = None,
    dayfirst: bool = False,
    complex_dates: bool = False,
) -> pd.Series:
    """
    Convert a raw string/numeric series to tz-aware UTC datetime.

    Resolution priority chain:
      1. complex_dates=True  -> dateutil.parser.parse
                                (handles "May 26, 2020, Tuesday" etc.)
      2. date_format set     -> exact strptime format  (e.g. '%Y%m%d')
      3. default             -> pandas mixed-format auto-detection with
                                dayfirst hint, then dateutil gap-fill
    """
    if complex_dates:
        return pd.to_datetime(
            series.map(_safe_dateutil_parse), errors="coerce", utc=True
        )

    if date_format:
        return pd.to_datetime(
            series.astype(str), format=date_format, errors="coerce", utc=True
        )

    # Standard mixed-format parse with dayfirst flag
    try:
        result = pd.to_datetime(
            series, format="mixed", dayfirst=dayfirst, errors="coerce", utc=True
        )
    except Exception:
        result = pd.to_datetime(series, errors="coerce", utc=True)

    # Second pass: fill any remaining NaT gaps with dateutil
    mask = result.isna()
    if mask.any():
        fallback = pd.to_datetime(
            series.where(~mask).map(_safe_dateutil_parse),
            errors="coerce",
            utc=True,
        )
        result = result.where(~mask, fallback)

    return result


# =============================================================================
# 8. GLOBAL POST-PROCESSING HELPERS
# =============================================================================

def _map_float_sentiment(
    df: pd.DataFrame,
    mapping: Dict[float, str],
    log: List[str],
) -> pd.DataFrame:
    """
    For each column whose name matches 'sentiment' or 'label', coerce values
    to float and replace with canonical strings via mapping.
    Non-matching numeric values are left as-is (cast to string).
    """
    for col in df.columns:
        if not re.search(r"\b(sentiment|label)\b", str(col), re.I):
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() == 0:
                continue
            mapped       = numeric.map(mapping)
            original_str = df[col].astype("string")
            # Keep original string where the float map has no entry
            df[col] = mapped.where(mapped.notna(), original_str)
            log.append(f"  sentiment_float_map applied -> column '{col}'")
        except Exception as exc:
            log.append(f"  WARN: sentiment_float_map skipped for '{col}': {exc}")
    return df


def _standardize_sentiment_strings(
    df: pd.DataFrame,
    target_cols: List[str],
) -> pd.DataFrame:
    """
    Title-case all string-based sentiment/label columns so that 'POSITIVE',
    'positive', and 'Positive' all become 'Positive'.
    """
    for col in target_cols:
        if (pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])):
            non_null_mask = df[col].notna()
            if non_null_mask.any():
                cleaned = df[col].copy()
                cleaned.loc[non_null_mask] = (
                    cleaned.loc[non_null_mask].astype(str).str.strip().str.title()
                )
                df[col] = cleaned
    return df


def _deduplicate(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Drop exact duplicate rows while protecting rows whose sentiment column
    is null.  Rows with a missing label are always preserved, because a
    missing annotation is meaningful rather than being a true duplicate.
    """
    if df.empty:
        return df
    if not target_cols:
        return df.drop_duplicates().reset_index(drop=True)

    missing_mask = pd.Series(False, index=df.index)
    for col in target_cols:
        col_null = df[col].isna()
        if (pd.api.types.is_object_dtype(df[col])
                or pd.api.types.is_string_dtype(df[col])):
            col_null = col_null | (
                df[col].astype("string").str.strip().str.lower()
                .isin(["", "nan", "none", "null"])
            )
        missing_mask = missing_mask | col_null

    deduped   = df.loc[~missing_mask].drop_duplicates()
    preserved = df.loc[missing_mask]
    return (
        pd.concat([deduped, preserved], axis=0)
        .sort_index(kind="stable")
        .reset_index(drop=True)
    )


# =============================================================================
# 9. PER-FILE PROCESSING  (executed inside worker processes)
# =============================================================================

def _process_file(path: Path, cfg: PreprocessConfig) -> ProcessResult:
    """
    Full pipeline for a single file:
      load -> drop_unnamed -> identify columns -> map float sentiment ->
      standardize dates -> normalize text -> title-case sentiment ->
      deduplicate -> write parquet

    All log messages are collected in-process and returned inside the result.
    The main process replays them into the log file.  This avoids concurrent
    writes to a shared FileHandler across multiple worker processes.
    """
    log: List[str] = []
    input_root  = Path(cfg.input_root).resolve()
    output_root = Path(cfg.output_root).resolve()

    # -- Resolve which FOLDER_CONFIGS entry governs this file -----------------
    folder_key: Optional[str] = None
    folder_cfg: Dict[str, Any] = {}
    for ancestor in path.resolve().parents:
        if ancestor.name in FOLDER_CONFIGS:
            folder_key = ancestor.name
            folder_cfg = FOLDER_CONFIGS[folder_key]
            log.append(f"FOLDER_CONFIG  '{folder_key}'")
            log.append(f"  active rules : {list(folder_cfg.keys())}")
            break

    log.append(f"START  {path}")

    try:
        # -- 1. Load ----------------------------------------------------------
        df = _apply_file_config(path, log, folder_cfg)
        file_cfg = FILE_CONFIGS.get(path.name.lower(), {})

        # -- 1a. File-level schema tweaks ------------------------------------
        rename_columns = file_cfg.get("rename_columns")
        if rename_columns:
            df = df.rename(columns=rename_columns)
            log.append(f"  file rename_columns applied: {rename_columns}")

        drop_columns = file_cfg.get("drop_columns")
        if drop_columns:
            df = df.drop(columns=drop_columns, errors="ignore")
            log.append(f"  file drop_columns applied: {drop_columns}")

        # Add symbol from filename for NIFTY 50 CSVs that do not provide one.
        parent_names = [ancestor.name.lower() for ancestor in path.resolve().parents]
        if "nifty 50" in parent_names and "Symbol" not in df.columns:
            df["Symbol"] = path.stem
            log.append(f"  symbol injected from filename: {path.stem}")

        # -- 1b. Schema standardization for downstream validator -------------
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            col_str = str(col).strip().lower()
            if col_str == "label":
                rename_map[col] = "sentiment"
            elif col_str == "publish_date" or col_str == "publish date":
                rename_map[col] = "date"
        if rename_map:
            df = df.rename(columns=rename_map)
            log.append(f"  schema standardized: {rename_map}")

        rows_in = len(df)
        log.append(
            f"  loaded  rows={rows_in}  cols={len(df.columns)}"
            f"  names={list(df.columns)}"
        )

        # -- 2. Drop Unnamed columns (Business News Headlines) ----------------
        if folder_cfg.get("drop_unnamed"):
            before  = list(df.columns)
            df      = df[[c for c in df.columns if not str(c).startswith("Unnamed:")]]
            dropped = [c for c in before if c not in df.columns]
            log.append(f"  drop_unnamed  removed={dropped}")

        skip_temporal_alignment = bool(file_cfg.get("skip_temporal_alignment", False))

        # -- 3. Identify semantic column roles --------------------------------
        date_col    = ColumnMapper.identify_date_column(df)
        text_cols   = ColumnMapper.identify_text_columns(df)
        target_cols = ColumnMapper.identify_target_columns(df)
        if skip_temporal_alignment:
            date_col = None
        log.append(
            f"  identified  date_col={date_col!r}"
            f"  text_cols={text_cols}"
            f"  target_cols={target_cols}"
        )
        if skip_temporal_alignment:
            log.append("  temporal alignment skipped (text-only dataset)")

        # -- 4. Float sentiment -> string mapping -----------------------------
        if folder_cfg.get("sentiment_float_map"):
            df = _map_float_sentiment(
                df, folder_cfg["sentiment_float_map"], log
            )

        # -- 5. Date standardization -> ISO 8601 ------------------------------
        date_format:   Optional[str] = folder_cfg.get("date_format")
        dayfirst:      bool          = bool(folder_cfg.get("dayfirst", False))
        complex_dates: bool          = bool(folder_cfg.get("complex_dates", False))

        if date_col is not None and date_col in df.columns:
            raw_non_null = df[date_col].notna()
            parsed    = _parse_datetime_series(
                df[date_col],
                date_format=date_format,
                dayfirst=dayfirst,
                complex_dates=complex_dates,
            )
            coerced_to_nat = int((raw_non_null & parsed.isna()).sum())
            valid_pct = parsed.notna().mean() * 100.0
            df[date_col] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
            mode_label = (
                date_format       if date_format   else
                "complex/dateutil" if complex_dates else
                f"mixed dayfirst={dayfirst}"
            )
            log.append(
                f"  date  col={date_col!r}  valid={valid_pct:.1f}%"
                f"  mode={mode_label}"
            )
            if coerced_to_nat > 0:
                log.append(
                    f"  info: date coercion -> NaT rows={coerced_to_nat}"
                    f"  col={date_col!r}"
                )
            if valid_pct < 50.0:
                log.append(
                    f"  WARN: date parse rate {valid_pct:.1f}% < 50% -"
                    " verify date_format in FOLDER_CONFIGS"
                )

        # -- 6. Text normalization (headline / title / text / content / description)
        # text_cols is pre-filtered by ColumnMapper to exclude sentiment/label cols.
        for col in text_cols:
            if col in df.columns:
                df[col] = TextNormalizer.clean_series(df[col])
                log.append(f"  text normalized: '{col}'")

        # -- 7. Sentiment string standardization (title-case) -----------------
        df = _standardize_sentiment_strings(df, target_cols)
        if target_cols:
            log.append(f"  sentiment title-cased: {target_cols}")

        # -- 8. Deduplication -------------------------------------------------
        df       = _deduplicate(df, target_cols)
        rows_out = len(df)
        log.append(f"  dedup  {rows_in} -> {rows_out} rows")

        # -- 9. Write parquet output ------------------------------------------
        rel      = path.resolve().relative_to(input_root)
        out_path = (output_root / rel).with_suffix(".parquet")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, engine="pyarrow")
        log.append(f"  wrote  -> {out_path}")
        log.append(f"OK     {path}")

        return ProcessResult(
            status="success",
            input_file=str(path),
            output_file=str(out_path),
            rows_in=rows_in,
            rows_out=rows_out,
            folder_config_key=folder_key,
            date_col=str(date_col) if date_col is not None else None,
            text_cols=text_cols,
            log_messages=log,
        )

    except Exception:
        tb = traceback.format_exc(limit=8)
        log.append(f"  ERROR:\n{tb}")
        log.append(f"FAIL   {path}")
        return ProcessResult(
            status="failed",
            input_file=str(path),
            output_file=None,
            rows_in=0,
            rows_out=0,
            folder_config_key=folder_key,
            date_col=None,
            text_cols=[],
            error=tb,
            log_messages=log,
        )


# =============================================================================
# 10. MULTIPROCESSING WORKER  (module-level required for pickling)
# =============================================================================

def _worker(file_path_str: str, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry-point for each worker process.  Receives plain-serializable args,
    reconstructs typed objects, and returns a plain dict so that
    ProcessPoolExecutor can pickle it back to the main process.
    """
    result = _process_file(Path(file_path_str), PreprocessConfig(**cfg_dict))
    return asdict(result)


# =============================================================================
# 11. FILE DISCOVERY
# =============================================================================

_SUPPORTED_EXTS: frozenset = frozenset({
    ".csv", ".tsv", ".txt", ".parquet",
    ".json", ".jsonl", ".xlsx", ".xls",
})

# Files that are documentation / scripts, not data - skip unconditionally.
_SKIP_SUFFIXES: frozenset = frozenset({
    ".py", ".md", ".rst", ".sh", ".bat", ".ipynb",
})
_SKIP_NAMES_LOWER: frozenset = frozenset({
    "readme.txt", "license.txt", "licence.txt", "readme.md",
    "changelog.txt", "requirements.txt",
})


def discover_files(root: Path) -> List[Path]:
    """
    Recursively find all processable data files under root.
    - Skips Python scripts, Markdown documentation, and similar non-data files.
    - Includes extensionless files (common for pre-processed TSV/CSV dumps).
    Returns a sorted list for deterministic ordering.
    """
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in _SKIP_SUFFIXES:
            continue
        if p.name.lower() in _SKIP_NAMES_LOWER:
            continue
        if p.suffix.lower() in _SUPPORTED_EXTS or p.suffix == "":
            files.append(p)
    return sorted(files)


# =============================================================================
# 12. LOGGING SETUP  (file ONLY - no StreamHandler, stdout stays clean)
# =============================================================================

def _setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("phase1_ingestion")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False  # Prevent leakage through root logger to stdout

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)
    return logger


# =============================================================================
# 13. PIPELINE ORCHESTRATION
# =============================================================================

def run_pipeline(cfg: PreprocessConfig) -> None:
    """
    Drive the full Phase 1 pipeline:
      discover files -> dispatch to worker pool -> collect results ->
      write log -> print terminal summary.
    """
    t_start     = time.monotonic()
    output_root = Path(cfg.output_root).resolve()
    if os.path.isdir(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    log_path    = output_root / cfg.log_file
    logger      = _setup_logger(log_path)

    input_root = Path(cfg.input_root).resolve()
    files      = discover_files(input_root)

    logger.info("=" * 72)
    logger.info("Phase 1 Ingestion Pipeline  -  STARTED")
    logger.info("=" * 72)
    logger.info("Input root   : %s", input_root)
    logger.info("Output root  : %s", output_root)
    logger.info(
        "Workers      : %d  (CPU cores available: %d)",
        cfg.workers, os.cpu_count() or 1,
    )
    logger.info("Files found  : %d", len(files))
    logger.info("-" * 72)
    logger.info("Active FOLDER_CONFIGS entries:")
    for k, v in FOLDER_CONFIGS.items():
        logger.info("  %-55s  rules=%s", repr(k), list(v.keys()))
    logger.info("-" * 72)

    if not files:
        logger.warning("No supported data files found. Exiting.")
        _print_summary(0, 0, 0, log_path, time.monotonic() - t_start)
        return

    max_workers = min(cfg.workers, os.cpu_count() or 1)
    cfg_dict    = asdict(cfg)
    success_n   = 0
    failed_n    = 0

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_worker, str(f), cfg_dict): f
            for f in files
        }
        for future in cf.as_completed(future_to_path):
            try:
                result = future.result()
            except Exception:
                # Worker process itself crashed (rare); log and continue.
                src = future_to_path[future]
                logger.error(
                    "WORKER CRASH | %s\n%s", src, traceback.format_exc()
                )
                failed_n += 1
                continue

            # Replay every message the worker collected
            for msg in result.get("log_messages", []):
                logger.debug("%s", msg)

            if result["status"] == "success":
                success_n += 1
                logger.info(
                    "OK   | rows %5d->%5d | cfg=%-50s | %s",
                    result["rows_in"],
                    result["rows_out"],
                    repr(result["folder_config_key"] or "-"),
                    result["input_file"],
                )
            else:
                failed_n += 1
                logger.error(
                    "FAIL | cfg=%s | %s\n%s",
                    repr(result["folder_config_key"] or "-"),
                    result["input_file"],
                    result.get("error", ""),
                )

    elapsed = time.monotonic() - t_start
    logger.info("=" * 72)
    logger.info(
        "COMPLETE | total=%d  success=%d  failed=%d  elapsed=%.2fs",
        len(files), success_n, failed_n, elapsed,
    )
    logger.info("=" * 72)

    _print_summary(len(files), success_n, failed_n, log_path, elapsed)


def _print_summary(
    total: int,
    success: int,
    failed: int,
    log_path: Path,
    elapsed: float,
) -> None:
    """
    The ONLY output that reaches the terminal (stdout).
    All detailed execution information is written exclusively to the log file.
    """
    bar = "=" * 56
    print(bar)
    print("  PHASE 1 INGESTION PIPELINE  -  COMPLETE")
    print(bar)
    print(f"  Total files   : {total}")
    print(f"  Successful    : {success}")
    print(f"  Failed        : {failed}")
    print(f"  Elapsed       : {elapsed:.2f}s")
    print(f"  Detailed log  : {log_path}")
    print(bar)


# =============================================================================
# 14. CLI ENTRY-POINT
# =============================================================================

def _parse_args() -> PreprocessConfig:
    p = argparse.ArgumentParser(
        description="Phase 1: configuration-driven financial data ingestion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-root",  default="datasets",
        help="Root directory containing the raw dataset folders",
    )
    p.add_argument(
        "--output-root", default="processed_phase1",
        help="Output directory for processed .parquet files",
    )
    p.add_argument(
        "--workers", type=int, default=24,
        help="Parallel worker processes (default: 24 for 24-core CPU)",
    )
    p.add_argument(
        "--log-file", default="phase1_ingestion.log",
        help="Log filename written inside --output-root",
    )
    args = p.parse_args()
    return PreprocessConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        workers=args.workers,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    run_pipeline(_parse_args())