# TMDB Movie Dataset Cleaning & Preprocessing with Python

A structured Python data cleaning pipeline for raw TMDB/IMDb-style movie metadata. This project transforms a 10,869-row, 21-column movie dataset into a clean, analysis-ready DataFrame by systematically resolving data type inconsistencies, sentinel value misuse, structural noise, and outliers. The end result is a reliable dataset ready for downstream analytics or machine learning workflows.

---

## Project Overview

Raw movie metadata sourced from TMDB/IMDb APIs typically arrives with mixed data types, zero-coded missing values, inconsistent date formats, and duplicate records — all of which silently corrupt aggregations and statistical analyses. This project implements a reproducible, step-by-step cleaning workflow in Python using pandas and NumPy to bring the dataset to a structurally sound and analytically trustworthy state.

The goal is not imputation or feature engineering, but rigorous data quality remediation: correcting types, exposing true missingness, standardising formats, and removing records that carry no informational value.

---

## Dataset Information

| Property | Details |
|---|---|
| Source | TMDB / IMDb-style movie metadata |
| Raw Shape | 10,869 rows × 21 columns |
| File | `Hey-pythonista.csv` |
| Output | `Hey-pythonista_cleaned.csv` |

**Key columns present in the dataset:**

| Column | Description |
|---|---|
| `id` | Numeric movie identifier |
| `imdb_id` | IMDb string identifier |
| `original_title` | Movie title |
| `release_date` | Release date (mixed formats in raw data) |
| `release_year` | Extracted year |
| `runtime` | Movie runtime in minutes |
| `budget` / `budget_adj` | Raw and inflation-adjusted budget |
| `revenue` / `revenue_adj` | Raw and inflation-adjusted revenue |
| `vote_average` | IMDb-scale rating (0–10) |
| `vote_count` | Number of votes |
| `popularity` | Popularity score |
| `genres` | Pipe-separated genre list |
| `cast` | Pipe-separated cast list |
| `director` | Director name |
| `keywords` | Pipe-separated keyword list |
| `production_companies` | Pipe-separated company list |
| `tagline` | Movie tagline |
| `overview` | Movie synopsis |

---

## Technologies Used

| Category | Tools |
|---|---|
| Programming Language | Python 3 |
| Data Processing | pandas, NumPy |
| Notebook Environment | Jupyter Notebook |
| Date Handling | `pd.to_datetime` |
| Warning Suppression | `warnings` (stdlib) |

---

## Data Cleaning Workflow

### Initial Dataset Inspection

The notebook begins with a structural audit of the raw dataset:

- `df.shape` to confirm raw dimensions (10,869 rows × 21 columns)
- `df.dtypes` and `df.info()` to surface type mismatches — notably `id`, `vote_count`, and `runtime` stored as `float64` despite being integer quantities
- `df.head()` for a visual scan of the first rows
- Null value inspection across all columns to inform treatment strategy

---

### Step 1 — Duplicate Removal

Identified and removed duplicate rows using `df.duplicated()`. Two exact duplicates were found and dropped with `drop_duplicates(inplace=True)`. Even a small number of duplicates inflates row counts and skews aggregations such as mean revenue and genre frequency.

---

### Step 2 — Drop All-Null Critical Rows

Rows where every column in the critical set — `id`, `imdb_id`, `original_title`, `release_date`, `runtime`, `vote_count`, `vote_average` — was simultaneously null were dropped. A conservative `how="all"` approach avoids over-dropping rows that are only partially missing, while ensuring that completely uninformative records do not persist in the dataset.

---

### Step 3 — Fix the `id` Column Type

The `id` column arrived as `float64` (e.g. `135397.0`) due to the presence of NaN values, which prevent standard `int` casting. It was converted to pandas nullable `Int64` using `pd.to_numeric(..., errors="coerce").astype("Int64")`. Storing identifiers as floats is semantically incorrect and risks silent precision loss on large integers.

---

### Step 4 — Standardise `release_date`

The `release_date` column contained a mix of Python `datetime` objects and date strings in formats like `"5/13/15"` and `"12/15/15"`. These were unified using `pd.to_datetime(..., errors="coerce")`, which coerces unparseable values to `NaT` rather than raising exceptions. The `release_year` column was then rebuilt from the cleaned `release_date` using `.dt.year.astype("Int64")` to ensure both columns remain consistent.

---

### Step 5 — Replace Zero Budgets and Revenues with NaN

Zeros in `budget`, `revenue`, `budget_adj`, and `revenue_adj` represent sentinel-coded missingness — not movies with zero budgets or zero earnings. These were replaced with `NaN` using `.replace(0, np.nan)`. Retaining zeros would severely distort financial summary statistics (mean, median) and any downstream visualisation or modelling.

---

### Step 6 — Cast `vote_count` and `runtime` to Integer

Both columns were stored as `float64` due to NaN propagation. They were converted to nullable `Int64` since fractional votes and fractional minutes are not meaningful. `popularity` and `vote_average` remain `float64` as fractional values are valid for those fields.

---

### Step 7 — Clean Pipe-Separated Text Columns

The columns `cast`, `genres`, `keywords`, and `production_companies` use `|` as a delimiter. A `clean_pipe_string()` function was applied to strip leading and trailing whitespace from each individual token (e.g. `"Action | Drama"` → `"Action|Drama"`). Without this step, the same category (e.g. `"Action"`) would appear as multiple distinct values during string splitting, groupby, or one-hot encoding.

---

### Step 8 — Strip Whitespace from String Columns

Leading and trailing whitespace was stripped from `imdb_id`, `original_title`, `director`, `tagline`, `overview`, and `homepage` using `.str.strip()`. Trailing spaces cause silent mismatches in equality checks and merges (e.g. `"Christopher Nolan"` ≠ `" Christopher Nolan"`).

---

### Step 9 — Replace Empty Strings with NaN

After stripping, some cells that previously contained only whitespace become empty strings (`""`). These are not captured by `.isnull()` and pollute value counts and analyses. A regex-based replacement (`df.replace(r"^\s*$", np.nan, regex=True)`) was applied across the entire DataFrame to unify all forms of missingness as `NaN`.

---

### Step 10 — Drop the `homepage` Column

The `homepage` column had 7,933 null values out of approximately 10,800 rows (~73% missing). At this sparsity level, the column contributes no meaningful signal to any typical movie analysis task and was dropped entirely to reduce noise and memory overhead.

---

### Step 11 — Remove `runtime` Outliers

Rows where `runtime` fell below 1 minute or exceeded 600 minutes (10 hours) were removed. Values outside this window are almost certainly data entry errors. The bounds are deliberately generous to preserve legitimate edge cases such as short films and extended epics, while excluding nonsensical entries that would distort distribution statistics.

---

### Step 12 — Remove Out-of-Range `vote_average` Values

IMDb ratings are bounded on the 0–10 scale. Rows where `vote_average` fell outside this range were removed as corrupt data that would distort rating-based filtering and analysis.

---

### Step 13 — Reset the Index

After all row-level drops, the DataFrame index contained gaps (e.g. `0, 1, 5, 9 …`). The index was reset with `reset_index(drop=True)` to produce a clean sequential index and prevent off-by-one errors in downstream row-based operations.

---

## Summary of Cleaning Operations

| Step | Issue | Action |
|---|---|---|
| 1 | 2 duplicate rows | Dropped |
| 2 | Rows with all critical fields null | Dropped |
| 3 | `id` stored as `float64` | Cast to nullable `Int64` |
| 4 | Mixed `release_date` formats | Unified to `datetime64`; rebuilt `release_year` |
| 5 | Zeros in budget/revenue used as "unknown" | Replaced with `NaN` |
| 6 | `vote_count`, `runtime` stored as `float64` | Cast to nullable `Int64` |
| 7 | Inconsistent spacing in pipe-delimited columns | Stripped per token |
| 8 | Leading/trailing whitespace in string columns | Stripped with `.str.strip()` |
| 9 | Empty strings after stripping | Replaced with `NaN` |
| 10 | `homepage` — 73% null | Column dropped |
| 11 | Impossible `runtime` values | Rows outside 1–600 min removed |
| 12 | `vote_average` outside 0–10 scale | Rows removed |
| 13 | Index gaps after row drops | Index reset |

---

## Project Structure

```bash
movie-data-cleaning/
│
├── data/
│   ├── Hey-pythonista.csv               # Raw dataset
│   └── Hey-pythonista_cleaned.csv       # Cleaned output
│
├── notebooks/
│   └── data_cleaning_python_notebook.ipynb
│
├── README.md
└── requirements.txt
```

---

## Installation & Setup

### Clone Repository

```bash
git clone <https://github.com/onyemarihappy/tmdb-movie-data-cleaning>
cd movie-data-cleaning
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Notebook

```bash
jupyter notebook
```

---

## Requirements

```txt
pandas
numpy
jupyter
```

---

## Final Output

The cleaned dataset (`Hey-pythonista_cleaned.csv`) reflects the following improvements over the raw input:

- **Structural correctness** — all columns carry semantically appropriate data types
- **Explicit missingness** — sentinel zeros and empty strings are unified as `NaN`, making null handling consistent across the entire DataFrame
- **Standardised dates** — `release_date` is a single `datetime64` type; `release_year` is consistent with it
- **Consistent categoricals** — pipe-separated columns are normalised and safe for token-level splitting and grouping
- **Clean string fields** — free-text columns are stripped of whitespace artifacts
- **Outlier-free numerics** — `runtime` and `vote_average` are bounded to valid real-world ranges
- **Compact schema** — low-value columns removed; index is sequential and gap-free

The dataset is ready for downstream analytics, statistical summarisation, or feature engineering for machine learning.

---

## Future Improvements

- Refactor the notebook into a modular Python script with configurable parameters for reuse across similar datasets
- Add a data validation layer (e.g. using `pandera` or `great_expectations`) to enforce schema contracts before and after cleaning
- Implement logging to capture row counts and null statistics at each cleaning step for auditability
- Extend the pipeline to support ingestion from a SQL database or object storage (S3/GCS) as a standalone ETL workflow
- Add unit tests for individual cleaning functions (e.g. `clean_pipe_string`) using `pytest`

---

## Author

Happiness Onyemari

- GitHub: github.com/onyemarihappy
