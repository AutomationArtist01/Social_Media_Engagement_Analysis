from __future__ import annotations
import os
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

import config
from utils import data_loader, eda
from utils.data_loader import DatasetNotFoundError
from utils.predictor import (
    ModelNotTrainedError,
    load_metrics,
    predict_engagement,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY


# ---------------------------------------------------------------------------
# Template context: make the current year and nav metadata available globally.
# ---------------------------------------------------------------------------
@app.context_processor
def inject_globals():
    return {
        "app_name": "SocialPulse",
        "year": 2026,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Landing page with a few headline numbers pulled from the dataset."""
    summary = None
    error = None
    try:
        df = data_loader.load_data()
        summary = data_loader.dataset_summary(df)
    except DatasetNotFoundError as exc:
        error = str(exc)
    return render_template("index.html", summary=summary, error=error)


@app.route("/dashboard")
def dashboard():
    """EDA dashboard: regenerate charts and render tables + insights."""
    try:
        df = data_loader.load_data()
    except DatasetNotFoundError as exc:
        return render_template("dashboard.html", error=str(exc))

    charts = eda.generate_all_charts(df)
    table = eda.platform_table(df)
    insights = eda.top_insights(df)
    summary = data_loader.dataset_summary(df)

    return render_template(
        "dashboard.html",
        charts=charts,
        table=table,
        insights=insights,
        summary=summary,
        error=None,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Show the prediction form (GET) or run a prediction (POST)."""
    result = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            result = predict_engagement(form_values)
        except ModelNotTrainedError as exc:
            error = str(exc)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "predict.html",
        result=result,
        error=error,
        form_values=form_values,
        platforms=config.PLATFORMS,
        post_types=config.POST_TYPES,
        topics=config.TOPICS,
        days=config.DAYS,
        sentiments=config.SENTIMENTS,
    )


@app.route("/performance")
def performance():
    """Display saved model metrics and feature importances."""
    metrics = load_metrics()
    return render_template("performance.html", metrics=metrics)


@app.route("/contact")
def contact():
    """Static contact page."""
    return render_template("contact.html")


# ---------------------------------------------------------------------------
# JSON API (bonus) — handy for testing with curl / other clients.
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Accept JSON or form data and return a JSON prediction."""
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({"error": "No input provided."}), 400
    try:
        result = predict_engagement(payload)
        return jsonify(result)
    except ModelNotTrainedError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return render_template("404.html"), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("500.html"), 500


if __name__ == "__main__":
    # Ensure the images directory exists so static charts can be written.
    os.makedirs(config.IMAGES_DIR, exist_ok=True)
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG)
