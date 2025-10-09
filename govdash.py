from __future__ import annotations

import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from arize.exporter import ArizeExportClient
from arize.utils.types import Environments

# -----------------------
# Styling
# -----------------------
FS_COLORS: Dict[str, str] = {
    "primary_blue": "#1B365D",
    "secondary_blue": "#2E5984",
    "accent_blue": "#4A90B8",
    "success_green": "#2E7D32",
    "warning_orange": "#F57C00",
    "danger_red": "#C62828",
    "neutral_gray": "#5F6368",
    "light_gray": "#F8F9FA",
    "white": "#FFFFFF",
    "text_dark": "#212529",
}

FS_PALETTE = [
    FS_COLORS["primary_blue"],
    FS_COLORS["secondary_blue"],
    FS_COLORS["accent_blue"],
    FS_COLORS["success_green"],
    FS_COLORS["warning_orange"],
    FS_COLORS["danger_red"],
]


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "grid.color": FS_COLORS["neutral_gray"],
    "grid.linewidth": 0.5,
    "axes.facecolor": FS_COLORS["white"],
    "figure.facecolor": FS_COLORS["white"],
    "text.color": FS_COLORS["text_dark"],
    "axes.labelcolor": FS_COLORS["text_dark"],
    "xtick.color": FS_COLORS["text_dark"],
    "ytick.color": FS_COLORS["text_dark"],
    "axes.edgecolor": FS_COLORS["neutral_gray"],
    "axes.linewidth": 1.0,
})

sns.set_palette(FS_PALETTE)

# -----------------------
# Logger
# -----------------------

def setup_logger(level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger("govdash")
    if not lg.handlers:
        lg.setLevel(level.upper())
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(h)
    return lg

# -----------------------
# Data helpers
# -----------------------

def pull_arize_data(api_key: str, space_id: str, model_id: str,
                    days_back: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    if ArizeExportClient is None:
        logger.warning("Arize SDK not installed; using sample data.")
        return None

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    logger.info(f"Pulling Arize data from {start_time.date()} to {end_time.date()}")
    try:
        client = ArizeExportClient(api_key=api_key)
        df = client.export_model_to_df(
            space_id=space_id,
            model_id=model_id,
            environment=Environments.TRACING,
            start_time=start_time,
            end_time=end_time,
            columns=[
                "context.span_id",
                "attributes.llm.model_name",
                "attributes.llm.provider",
                "attributes.llm.token_count.total",
                "attributes.llm.token_count.prompt",
                "attributes.llm.token_count.completion",
                "status_code",
                "start_time",
                "end_time",
                "name",
            ],
        )
        logger.info(f"Pulled {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Arize pull failed: {e}")
        return None


def sample_data(days_back: int, logger: logging.Logger, n: int = 380) -> pd.DataFrame:
    logger.info("Using deterministic sample data fallback")
    rng = np.random.default_rng(7)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    dates = pd.date_range(start_time, end_time, periods=n)

    df = pd.DataFrame({
        "context.span_id": [f"span_{i:04d}" for i in range(n)],
        "attributes.llm.model_name": rng.choice(["gpt-4", "gpt-3.5-turbo", "claude-3", "gpt-4o"], n),
        "attributes.llm.provider": rng.choice(["openai", "anthropic"], n),
        "attributes.llm.token_count.total": rng.integers(50, 3200, n),
        "status_code": rng.choice(["OK", "ERROR"], n, p=[0.94, 0.06]),
        "start_time": dates,
        "end_time": dates + pd.to_timedelta(rng.integers(1, 28, n), unit="s"),
        "name": ["ChatCompletion"] * n,
    })
    return df


def clean(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"])  # type: ignore
    df["end_time"] = pd.to_datetime(df["end_time"])      # type: ignore
    df["hour"] = df["end_time"].dt.hour
    df["date"] = df["end_time"].dt.date
    df["day_of_week"] = df["end_time"].dt.day_name()
    df["duration_s"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
    df["attributes.llm.token_count.total"] = df["attributes.llm.token_count.total"].fillna(0)
    df["status_code"] = df["status_code"].fillna("UNKNOWN")
    logger.info(f"Cleaned data shape: {df.shape}")
    return df

# -----------------------
# Simple proxy metrics (for demo visuals only)
# -----------------------

def kpis(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "total_requests": float(len(df)),
        "success_rate": float((df["status_code"].eq("OK")).mean() * 100),
        "total_tokens": float(df["attributes.llm.token_count.total"].sum()),
        "avg_tokens": float(df["attributes.llm.token_count.total"].mean()),
        "active_models": float(df["attributes.llm.model_name"].nunique()),
        "avg_quality": float(  # 0-100 proxy based on short duration + non-error
            (100 * (1 - (df["duration_s"].clip(0, 30) / 30)).mean() * 0.5
             + (df["status_code"].eq("OK")).mean() * 50)
        ),
        "peak_hour": float(df.groupby("hour").size().idxmax()) if len(df) else 0.0,
    }


def proxy_response_quality(df: pd.DataFrame) -> Dict[str, float]:
    # Deterministic proxies for demo visuals
    dur = 1 - (df["duration_s"].clip(0, 30) / 30)
    ok = df["status_code"].eq("OK").astype(float)
    tok = 1 - (df["attributes.llm.token_count.total"].clip(0, 2500) / 2500)
    return {
        "Coherence": float((0.6 * dur + 0.4 * ok).mean() * 100),
        "Relevance": float((0.5 * tok + 0.5 * ok).mean() * 100),
        "Helpfulness": float((0.5 * dur + 0.5 * tok).mean() * 100),
    }


def proxy_bias_groups(df: pd.DataFrame) -> Dict[str, float]:
    # Use provider as two groups and compare mean proxy quality
    q = proxy_response_quality(df)
    base = (q["Coherence"] + q["Relevance"] + q["Helpfulness"]) / 3

    means = df.groupby("attributes.llm.provider")["attributes.llm.token_count.total"].mean()
    if len(means) < 2:
        return {"Group A": base, "Group B": base}
    # Map higher token usage to slightly lower score to simulate disparity
    mn, mx = means.min(), means.max()
    if mx == mn:
        return {"Group A": base, "Group B": base}
    gap = (mx - mn) / mx  # 0..1
    return {"Group A": base, "Group B": max(0.0, base * (1 - 0.2 * gap))}

# -----------------------
# Plots
# -----------------------

def kpi_band(ax: plt.Axes, df: pd.DataFrame) -> None:
    m = kpis(df)
    items = [
        ("TOTAL REQUESTS", f"{int(m['total_requests']):,}"),
        ("SUCCESS RATE", f"{m['success_rate']:.1f}%"),
        ("TOTAL TOKENS", f"{int(m['total_tokens']):,}"),
        ("AVG TOKENS/REQ", f"{m['avg_tokens']:.0f}"),
        ("ACTIVE MODELS", f"{int(m['active_models']):d}"),
        ("PEAK HOUR", f"{int(m['peak_hour'])}:00"),
        ("AVG QUALITY", f"{m['avg_quality']:.1f}%"),
    ]
    ax.axis("off")
    x0, dx = 0.015, 0.14
    for i, (label, value) in enumerate(items):
        ax.text(x0 + i * dx, 0.68, label, transform=ax.transAxes,
                fontsize=9.5, color=FS_COLORS["neutral_gray"]) 
        ax.text(x0 + i * dx, 0.30, value, transform=ax.transAxes,
                fontsize=16, fontweight="bold", color=FS_COLORS["primary_blue"]) 


def pie_model_usage(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts = df["attributes.llm.model_name"].value_counts()
    ax.clear()
    if counts.empty:
        ax.axis("off"); ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    colors = FS_PALETTE[: len(counts)]
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_color("white"); t.set_fontweight("bold")
    ax.set_title("Model Usage Distribution", loc="left", color=FS_COLORS["primary_blue"]) 


def line_tokens_by_hour(ax: plt.Axes, df: pd.DataFrame) -> None:
    hourly = df.groupby("hour")["attributes.llm.token_count.total"].sum()
    ax.plot(hourly.index, hourly.values, marker="o", linewidth=2)
    ax.fill_between(hourly.index, hourly.values, alpha=0.12)
    ax.set_title("Token Usage by Hour", loc="left", color=FS_COLORS["primary_blue"]) 
    ax.set_xlabel("Hour"); ax.set_ylabel("Total Tokens")


def bar_system_health(ax: plt.Axes, df: pd.DataFrame) -> None:
    counts = df["status_code"].value_counts()
    colors = [FS_COLORS["success_green"] if k == "OK" else FS_COLORS["danger_red"] for k in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, alpha=0.9, edgecolor="white", linewidth=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.01, f"{int(b.get_height())}",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("System Health Status", loc="left", color=FS_COLORS["primary_blue"]) 
    ax.set_ylabel("Requests")


def hist_tokens(ax: plt.Axes, df: pd.DataFrame) -> None:
    vals = df.loc[df["attributes.llm.token_count.total"] > 0, "attributes.llm.token_count.total"]
    ax.hist(vals, bins=20, alpha=0.8, edgecolor="white")
    mean_v = float(vals.mean()) if len(vals) else 0
    ax.axvline(mean_v, linestyle="--", linewidth=2, label=f"Mean: {mean_v:.0f}")
    ax.legend(loc="upper right")
    ax.set_title("Token Usage Distribution", loc="left", color=FS_COLORS["primary_blue"]) 
    ax.set_xlabel("Tokens/Request"); ax.set_ylabel("Frequency")


def bars_response_quality(ax: plt.Axes, df: pd.DataFrame) -> None:
    scores = proxy_response_quality(df)
    labels = list(scores.keys()); values = [scores[k] for k in labels]
    bars = ax.bar(labels, values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Response Quality (proxy)", loc="left", color=FS_COLORS["primary_blue"]) 
    ax.set_ylabel("Average score (%)")


def bars_bias_detection(ax: plt.Axes, df: pd.DataFrame) -> None:
    grp = proxy_bias_groups(df)
    labels = list(grp.keys()); values = [grp[k] for k in labels]
    bars = ax.bar(labels, values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title("Bias Detection (proxy parity)", loc="left", color=FS_COLORS["primary_blue"]) 
    ax.set_ylabel("Score (%)")

# -----------------------
# Figure builder
# -----------------------

def build_dashboard(df: pd.DataFrame) -> plt.Figure:
    mosaic = [
        ["T", "T", "T", "T"],
        ["B", "B", "B", "B"],
        ["P", "L", "H", "H"],
        ["D", "Q", "X", "X"],
    ]
    # Rows: Title, KPIs, (Pie/Line/Health), (Dist/Quality/Bias)
    height_ratios = [0.11, 0.15, 0.36, 0.38]

    fig, axs = plt.subplot_mosaic(
        mosaic,
        figsize=(22, 14),
        constrained_layout=False,
        gridspec_kw={"height_ratios": height_ratios, "wspace": 0.24, "hspace": 0.34},
    )
    fig.patch.set_facecolor(FS_COLORS["white"])

    # Title
    axs["T"].axis("off")
    axs["T"].text(
        0.01,
        0.60,
        "AI GOVERNANCE & RISK MANAGEMENT DASHBOARD",
        fontsize=22,
        fontweight="bold",
        color=FS_COLORS["primary_blue"],
        transform=axs["T"].transAxes,
    )
    axs["T"].text(
        0.01,
        0.22,
        "Real-time Monitoring & Compliance Assessment",
        fontsize=12,
        color=FS_COLORS["neutral_gray"],
        style="italic",
        transform=axs["T"].transAxes,
    )

    # KPI band
    kpi_band(axs["B"], df)

    # Charts
    pie_model_usage(axs["P"], df)
    line_tokens_by_hour(axs["L"], df)
    bar_system_health(axs["H"], df)
    hist_tokens(axs["D"], df)
    bars_response_quality(axs["Q"], df)
    bars_bias_detection(axs["X"], df)

    fig.subplots_adjust(left=0.04, right=0.985, top=0.97, bottom=0.05, wspace=0.24, hspace=0.34)
    return fig

# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AI Governance dashboard (lite)")
    p.add_argument("--api-key", default=os.getenv("ARIZE_API_KEY", ""))
    p.add_argument("--space-id", default=os.getenv("ARIZE_SPACE_ID", ""))
    p.add_argument("--model-id", default=os.getenv("ARIZE_PROJECT_NAME", ""))
    p.add_argument("--days-back", type=int, default=int(os.getenv("DAYS_BACK", "7")))
    p.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", "/mnt/artifacts")))
    p.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logger(args.log_level)

    # Pull or fall back
    df: Optional[pd.DataFrame]
    if args.api_key and args.space_id and args.model_id:
        df = pull_arize_data(args.api_key, args.space_id, args.model_id, args.days_back, log)
    else:
        log.warning("Missing Arize creds; falling back to sample data.")
        df = None

    if df is None or df.empty:
        print("didnt find any data. check api keys in env variables")
        exit()

    df = clean(df, log)

    fig = build_dashboard(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "ai_governance_dashboard_lite.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.3)
    log.info(f"Saved dashboard: {out}")

    # Uncomment to view interactively
    # plt.show()


if __name__ == "__main__":
    main()
