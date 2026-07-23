import os
import unittest
import config
from app import app
from utils import data_loader
from utils.predictor import predict_engagement


class DataPipelineTests(unittest.TestCase):
    def test_dataset_exists(self):
        self.assertTrue(
            os.path.exists(config.DATASET_PATH),
            "Dataset missing — run `python generate_dataset.py` first.",
        )

    def test_load_and_clean(self):
        df = data_loader.load_data()
        self.assertGreater(len(df), 0)
        for col in config.CATEGORICAL_FEATURES + config.NUMERIC_FEATURES:
            self.assertIn(col, df.columns)
        # Target must exist and have no NaNs after cleaning.
        self.assertFalse(df[config.TARGET].isna().any())

    def test_summary_keys(self):
        df = data_loader.load_data()
        summary = data_loader.dataset_summary(df)
        for key in ("total_posts", "avg_engagement", "platforms", "best_platform"):
            self.assertIn(key, summary)


class PredictionTests(unittest.TestCase):
    def test_valid_prediction(self):
        sample = {
            "platform": "Instagram",
            "post_type": "Reel",
            "topic": "Travel",
            "day_of_week": "Saturday",
            "sentiment": "Positive",
            "post_hour": 19,
            "followers": 25000,
            "hashtags_count": 10,
            "caption_length": 120,
        }
        result = predict_engagement(sample)
        self.assertIn("engagement_rate", result)
        self.assertGreaterEqual(result["engagement_rate"], 0.0)
        self.assertIn(result["band"], ("Low", "Medium", "High"))

    def test_missing_field_raises(self):
        with self.assertRaises(ValueError):
            predict_engagement({"platform": "Instagram"})

    def test_out_of_range_raises(self):
        sample = {
            "platform": "Instagram",
            "post_type": "Reel",
            "topic": "Travel",
            "day_of_week": "Saturday",
            "sentiment": "Positive",
            "post_hour": 99,  # invalid
            "followers": 25000,
            "hashtags_count": 10,
            "caption_length": 120,
        }
        with self.assertRaises(ValueError):
            predict_engagement(sample)


class RouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_home(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_dashboard(self):
        self.assertEqual(self.client.get("/dashboard").status_code, 200)

    def test_predict_get(self):
        self.assertEqual(self.client.get("/predict").status_code, 200)

    def test_predict_post(self):
        resp = self.client.post(
            "/predict",
            data={
                "platform": "TikTok",
                "post_type": "Video",
                "topic": "Fitness",
                "day_of_week": "Friday",
                "sentiment": "Positive",
                "post_hour": 20,
                "followers": 50000,
                "hashtags_count": 8,
                "caption_length": 90,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Predicted Engagement Rate", resp.data)

    def test_performance(self):
        self.assertEqual(self.client.get("/performance").status_code, 200)

    def test_contact(self):
        self.assertEqual(self.client.get("/contact").status_code, 200)

    def test_api_predict(self):
        resp = self.client.post(
            "/api/predict",
            json={
                "platform": "YouTube",
                "post_type": "Video",
                "topic": "Tech",
                "day_of_week": "Monday",
                "sentiment": "Neutral",
                "post_hour": 12,
                "followers": 100000,
                "hashtags_count": 5,
                "caption_length": 200,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("engagement_rate", resp.get_json())

    def test_404(self):
        self.assertEqual(self.client.get("/does-not-exist").status_code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
