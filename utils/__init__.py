"""Utility package for the Social Media Engagement Analysis project.

Modules
-------
data_loader : loading and light cleaning of the dataset
eda         : exploratory data analysis + chart generation
predictor   : load the trained model and make predictions
"""

from . import data_loader, eda, predictor  # noqa: F401

__all__ = ["data_loader", "eda", "predictor"]
