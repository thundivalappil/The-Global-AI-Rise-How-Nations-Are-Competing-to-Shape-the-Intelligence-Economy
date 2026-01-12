"""
Global AI Raise Visualizations
Creates a set of charts you can use in articles and presentations:
1) Global AI private investment trend (World if available)
2) Top entities (countries/regions) by latest-year AI investment
3) Top-N entity trends (lines)
4) Country AI strengths heatmap (editable heuristic scores)

Data source for investment trend: Our World in Data (OWID) grapher CSV.
If you are offline, download the CSV once and set OWID_SOURCE to your local file path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# USER SETTINGS
# -----------------------------
OWID_SOURCE = "https://ourworldindata.org/grapher/private-investment-in-artificial-intelligence.csv"
OUTPUT_DIR = Path("ai_outputs")
SAVE_PNG = True
SHOW_PLOTS = True

TOP_N_BAR = 15          # Top entities in latest-year bar chart
TOP_N_TRENDS = 6        # Top entities to show in multi-line trend chart
EXCLUDE_ENTITIES = {"World"}  # Exclude from "Top entities" charts (keep for global trend)


# -----------------------------
# UTILITIES
# -----------------------------
def _ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _savefig(filename: str) -> None:
    if SAVE_PNG:
        _ensure_output_dir()
        out = OUTPUT_DIR / filename
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {out.resolve()}")


def load_owid_csv(source: str) -> pd.DataFrame:
    """
    Load OWID grapher CSV either from a URL or a local file path.
    Expected columns: Entity, Code, Year, <metric>
    """
    try:
        df = pd.read_csv(source)
        return df
    except Exception as e:
        raise RuntimeError(
            "Could not load OWID data.\n"
            f"- Source: {source}\n"
            "If you are offline, download the CSV from OWID and set OWID_SOURCE to the local file path.\n"
            f"Original error: {e}"
        ) from e


def detect_metric_column(df: pd.DataFrame) -> str:
    id_cols = {"Entity", "Code", "Year"}
    metric_cols = [c for c in df.columns if c not in id_cols]
    if not metric_cols:
        raise ValueError(f"Metric column not found. Columns: {list(df.columns)}")
    if len(metric_cols) > 1:
        # OWID grapher usually has just one metric column; pick the first, but warn.
        print(f"⚠️ Multiple metric columns found; using '{metric_cols[0]}': {metric_cols}")
    return metric_cols[0]


def latest_year(df: pd.DataFrame) -> int:
    if "Year" not in df.columns:
        raise ValueError("Year column missing.")
    return int(pd.to_numeric(df["Year"], errors="coerce").dropna().max())


def world_series_or_sum(df: pd.DataFrame, metric: str) -> Tuple[pd.DataFrame, str]:
    """
    Prefer 'World' time series if present; otherwise sum across entities by year (may double-count).
    Returns (dataframe with Year, metric), suffix string for title.
    """
    if (df["Entity"] == "World").any():
        g = (
            df.loc[df["Entity"] == "World", ["Year", metric]]
            .dropna()
            .sort_values("Year")
        )
        return g, " (World)"
    g = (
        df.groupby("Year", as_index=False)[metric]
        .sum(min_count=1)
        .dropna()
        .sort_values("Year")
    )
    return g, " (sum across entities)"


def clean_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# -----------------------------
# CHARTS
# -----------------------------
def plot_global_trend(df: pd.DataFrame, metric: str) -> None:
    g, title_suffix = world_series_or_sum(df, metric)
    g["Year"] = clean_numeric_series(g["Year"])
    g[metric] = clean_numeric_series(g[metric])

    plt.figure(figsize=(10, 5))
    plt.plot(g["Year"], g[metric])
    plt.title(f"Global Private Investment in AI — Trend{title_suffix}")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.tight_layout()
    _savefig("01_global_ai_investment_trend.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_top_entities_latest_year(df: pd.DataFrame, metric: str, top_n: int = TOP_N_BAR) -> None:
    yr = latest_year(df)
    d = df.copy()
    d["Year"] = clean_numeric_series(d["Year"])
    d[metric] = clean_numeric_series(d[metric])

    latest = d.loc[d["Year"] == yr, ["Entity", metric]].dropna()

    # Exclude generic aggregates if desired
    if EXCLUDE_ENTITIES:
        latest = latest.loc[~latest["Entity"].isin(EXCLUDE_ENTITIES)]

    latest = latest.sort_values(metric, ascending=False).head(top_n)
    latest = latest.sort_values(metric, ascending=True)  # horizontal bar nicer

    plt.figure(figsize=(10, 6))
    plt.barh(latest["Entity"], latest[metric])
    plt.title(f"Top {top_n} Entities by AI Private Investment ({yr})")
    plt.xlabel(metric)
    plt.ylabel("Entity")
    plt.tight_layout()
    _savefig("02_top_entities_latest_year.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_top_entity_trends(df: pd.DataFrame, metric: str, top_n: int = TOP_N_TRENDS) -> None:
    """
    Pick top-N entities by latest-year value and plot their time series lines.
    """
    yr = latest_year(df)
    d = df.copy()
    d["Year"] = clean_numeric_series(d["Year"])
    d[metric] = clean_numeric_series(d[metric])

    latest = d.loc[d["Year"] == yr, ["Entity", metric]].dropna()
    if EXCLUDE_ENTITIES:
        latest = latest.loc[~latest["Entity"].isin(EXCLUDE_ENTITIES)]
    top_entities = (
        latest.sort_values(metric, ascending=False)["Entity"].head(top_n).tolist()
    )
    if not top_entities:
        print("⚠️ Could not determine top entities for trend plot.")
        return

    plt.figure(figsize=(11, 6))
    for ent in top_entities:
        series = (
            d.loc[d["Entity"] == ent, ["Year", metric]]
            .dropna()
            .sort_values("Year")
        )
        if series.empty:
            continue
        plt.plot(series["Year"], series[metric], label=ent)

    plt.title(f"AI Private Investment — Trends for Top {len(top_entities)} Entities")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.legend(loc="best")
    plt.tight_layout()
    _savefig("03_top_entity_trends.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_country_expertise_heatmap() -> None:
    """
    Heuristic, editable scores (1..5) to visualize country capability areas.
    This chart is independent of the OWID investment dataset.
    """
    countries: List[str] = [
        "United States", "China", "United Kingdom", "France", "Singapore",
        "South Korea", "Israel", "Canada", "India", "Japan", "Germany"
    ]

    # 1 (lower) .. 5 (higher). Editable heuristic scores.
    data = {
        "Talent":                 [5, 4, 4, 4, 4, 4, 4, 4, 5, 4, 4],
        "Research":               [5, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4],
        "Development":            [5, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4],
        "Infrastructure/Compute": [5, 4, 4, 4, 4, 5, 4, 4, 3, 4, 4],
        "Commercial Ecosystem":   [5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        "Gov Strategy/Policy":    [4, 4, 5, 5, 4, 4, 4, 4, 4, 4, 4],
    }

    df_heat = pd.DataFrame(data, index=countries).astype(float)

    plt.figure(figsize=(12, 6))
    plt.imshow(df_heat.values, aspect="auto")
    plt.title("Leading Countries — AI Strength Areas (Editable Heatmap)")
    plt.xlabel("AI capability areas")
    plt.ylabel("Country")
    plt.xticks(range(len(df_heat.columns)), df_heat.columns, rotation=25, ha="right")
    plt.yticks(range(len(df_heat.index)), df_heat.index)

    for i in range(df_heat.shape[0]):
        for j in range(df_heat.shape[1]):
            plt.text(j, i, int(df_heat.iat[i, j]), ha="center", va="center", fontsize=9)

    plt.colorbar(label="Score (1–5)")
    plt.tight_layout()
    _savefig("04_country_strengths_heatmap.png")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
def main() -> int:
    print("=== Global AI Raise Visualizations ===")

    # 1) Investment charts from OWID
    df = load_owid_csv(OWID_SOURCE)
    metric = detect_metric_column(df)
    print(f"✅ Loaded OWID dataset. Metric column: {metric}. Rows: {len(df):,}")

    plot_global_trend(df, metric)
    plot_top_entities_latest_year(df, metric, top_n=TOP_N_BAR)
    plot_top_entity_trends(df, metric, top_n=TOP_N_TRENDS)

    # 2) Expertise heatmap (independent)
    plot_country_expertise_heatmap()

    print("✅ Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
