"""
config.py
=========
Central configuration for the Social Media Engagement Analysis project.
Keeping paths and constants in one place avoids magic strings scattered
across the codebase.
"""

import os

# Absolute path to the project root (directory that contains this file).
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Data ----------------------------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "social_media_data.csv")

# Models --------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "engagement_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

# Static assets (generated EDA charts land here) ----------------------------
STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

# Flask ---------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
# Port can be overridden via the PORT env var (macOS reserves 5000 for AirPlay).
PORT = int(os.environ.get("PORT", "5000"))

# Feature configuration -----------------------------------------------------
# Columns the model treats as categorical vs. numeric. The target is what we
# predict. These lists are consumed by the training and prediction code so
# that both stay perfectly in sync.
CATEGORICAL_FEATURES = [
    "platform",
    "post_type",
    "topic",
    "day_of_week",
    "sentiment",
]
NUMERIC_FEATURES = [
    "is_weekend",
    "post_hour",
    "followers",
    "hashtags_count",
    "caption_length",
]
TARGET = "engagement_rate"

# Allowed category values (used to populate dropdowns and validate input).
PLATFORMS = ["Instagram", "Facebook", "Twitter", "LinkedIn", "TikTok", "YouTube"]
POST_TYPES = ["Image", "Video", "Carousel", "Text", "Reel", "Story"]
TOPICS = ["Tech", "Fashion", "Food", "Travel", "Fitness", "Business", "Entertainment"]
DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

# Engagement-level thresholds (percent) used to turn a numeric prediction
# into a friendly Low / Medium / High label.
ENGAGEMENT_BANDS = [
    (0.0, 1.5, "Low"),
    (1.5, 3.5, "Medium"),
    (3.5, float("inf"), "High"),
]


def band_for_rate(rate: float) -> str:
    """Return the engagement band label ('Low'/'Medium'/'High') for a rate."""
    for low, high, label in ENGAGEMENT_BANDS:
        if low <= rate < high:
            return label
    return "Unknown"
