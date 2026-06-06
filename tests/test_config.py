"""Smoke tests for configuration module."""

from jse_radar.config import ROOT_DIR, RAW_DIR, PROCESSED_DIR


def test_root_dir_exists():
    assert ROOT_DIR.exists(), f"ROOT_DIR does not exist: {ROOT_DIR}"


def test_data_dirs_are_pathlib():
    assert hasattr(RAW_DIR, "mkdir")
    assert hasattr(PROCESSED_DIR, "mkdir")