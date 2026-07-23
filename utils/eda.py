from __future__ import annotations
import os
import matplotlib

# Use a non-interactive backend so charts render on a headless server.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402

# Consistent, pleasant theme across every chart.
sns.set_theme(style="whitegrid")
PALETTE = "viridis"
FIGSIZE = (8, 5)


def _save(fig, filename: str) -> str:
    """Save ``fig`` to the images dir and return the static-relative path."""
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    out_path = os.path.join(config.IMAGES_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    # Path used by templates via url_for('static', filename=...).
    return f"images/{filename}"


# ---------------------------------------------------------------------------
# Individual charts
# ---------------------------------------------------------------------------
def chart_engagement_by_platform(df: pd.DataFrame) -> str:
    grp = (
        df.groupby("platform")[config.TARGET]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=grp, x="platform", y=config.TARGET, hue="platform",
        palette=PALETTE, legend=False, ax=ax,
    )
    ax.set_title("Average Engagement Rate by Platform")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Avg Engagement Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, "engagement_by_platform.png")


def chart_engagement_by_post_type(df: pd.DataFrame) -> str:
    grp = (
        df.groupby("post_type")[config.TARGET]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=grp, x="post_type", y=config.TARGET, hue="post_type",
        palette="magma", legend=False, ax=ax,
    )
    ax.set_title("Average Engagement Rate by Post Type")
    ax.set_xlabel("Post Type")
    ax.set_ylabel("Avg Engagement Rate (%)")
    return _save(fig, "engagement_by_post_type.png")


def chart_engagement_by_hour(df: pd.DataFrame) -> str:
    grp = df.groupby("post_hour")[config.TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(
        data=grp, x="post_hour", y=config.TARGET, marker="o", color="#6c5ce7", ax=ax
    )
    ax.set_title("Engagement Rate by Hour of Day")
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Avg Engagement Rate (%)")
    ax.set_xticks(range(0, 24, 2))
    return _save(fig, "engagement_by_hour.png")


def chart_engagement_by_day(df: pd.DataFrame) -> str:
    order = config.DAYS
    grp = df.groupby("day_of_week")[config.TARGET].mean().reindex(order).reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=grp, x="day_of_week", y=config.TARGET, hue="day_of_week",
        palette="crest", legend=False, ax=ax,
    )
    ax.set_title("Average Engagement Rate by Day of Week")
    ax.set_xlabel("Day")
    ax.set_ylabel("Avg Engagement Rate (%)")
    ax.tick_params(axis="x", rotation=20)
    return _save(fig, "engagement_by_day.png")


def chart_engagement_distribution(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.histplot(df[config.TARGET], bins=40, kde=True, color="#00b894", ax=ax)
    ax.set_title("Distribution of Engagement Rate")
    ax.set_xlabel("Engagement Rate (%)")
    ax.set_ylabel("Number of Posts")
    return _save(fig, "engagement_distribution.png")


def chart_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = df[config.NUMERIC_FEATURES + ["likes", "comments", "shares", config.TARGET]]
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Correlation Heatmap of Numeric Features")
    return _save(fig, "correlation_heatmap.png")


def chart_sentiment_pie(df: pd.DataFrame) -> str:
    counts = df["sentiment"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette("pastel")[: len(counts)]
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title("Post Sentiment Breakdown")
    return _save(fig, "sentiment_pie.png")


def chart_hashtags_vs_engagement(df: pd.DataFrame) -> str:
    grp = df.groupby("hashtags_count")[config.TARGET].mean().reset_index()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.lineplot(
        data=grp, x="hashtags_count", y=config.TARGET, marker="o", color="#e17055", ax=ax
    )
    ax.set_title("Hashtag Count vs Engagement Rate")
    ax.set_xlabel("Number of Hashtags")
    ax.set_ylabel("Avg Engagement Rate (%)")
    return _save(fig, "hashtags_vs_engagement.png")


# ---------------------------------------------------------------------------
# Orchestration + tabular summaries
# ---------------------------------------------------------------------------
def generate_all_charts(df: pd.DataFrame) -> dict:
    """Generate every chart and return a name->static-path mapping."""
    return {
        "engagement_by_platform": chart_engagement_by_platform(df),
        "engagement_by_post_type": chart_engagement_by_post_type(df),
        "engagement_by_hour": chart_engagement_by_hour(df),
        "engagement_by_day": chart_engagement_by_day(df),
        "engagement_distribution": chart_engagement_distribution(df),
        "correlation_heatmap": chart_correlation_heatmap(df),
        "sentiment_pie": chart_sentiment_pie(df),
        "hashtags_vs_engagement": chart_hashtags_vs_engagement(df),
    }


def platform_table(df: pd.DataFrame) -> list[dict]:
    """Aggregated per-platform stats for a dashboard table."""
    grp = (
        df.groupby("platform")
        .agg(
            posts=("engagement_rate", "size"),
            avg_engagement=("engagement_rate", "mean"),
            avg_likes=("likes", "mean"),
            avg_followers=("followers", "mean"),
        )
        .sort_values("avg_engagement", ascending=False)
        .reset_index()
    )
    grp["avg_engagement"] = grp["avg_engagement"].round(2)
    grp["avg_likes"] = grp["avg_likes"].round(0).astype(int)
    grp["avg_followers"] = grp["avg_followers"].round(0).astype(int)
    return grp.to_dict(orient="records")


def top_insights(df: pd.DataFrame) -> list[str]:
    """Human-readable bullet insights derived from the data."""
    best_platform = df.groupby("platform")[config.TARGET].mean().idxmax()
    best_type = df.groupby("post_type")[config.TARGET].mean().idxmax()
    best_hour = int(df.groupby("post_hour")[config.TARGET].mean().idxmax())
    best_day = df.groupby("day_of_week")[config.TARGET].mean().idxmax()
    weekend_lift = (
        df[df["is_weekend"] == 1][config.TARGET].mean()
        - df[df["is_weekend"] == 0][config.TARGET].mean()
    )
    return [
        f"'{best_platform}' delivers the highest average engagement rate.",
        f"'{best_type}' posts outperform every other content format.",
        f"Posting around {best_hour:02d}:00 tends to maximise engagement.",
        f"{best_day} is statistically the strongest day to post.",
        (
            "Weekend posts engage "
            f"{'better' if weekend_lift > 0 else 'worse'} "
            f"by {abs(weekend_lift):.2f} percentage points on average."
        ),
    ]
