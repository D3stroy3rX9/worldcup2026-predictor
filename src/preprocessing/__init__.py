"""Preprocessing module for World Cup 2026 Predictor."""

from .feature_engineering import FeatureEngineer, main as build_features

__all__ = ['FeatureEngineer', 'build_features']
