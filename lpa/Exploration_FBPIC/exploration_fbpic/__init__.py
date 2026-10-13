"""Staged Bayesian exploration campaigns for FBPIC ionization injection."""

__version__ = "0.1.0"

from .campaign import Campaign, CampaignConfig, Parameter

__all__ = ["Campaign", "CampaignConfig", "Parameter", "__version__"]
