"""Shared pytest fixtures for dopest-clip.

Tests that need ffmpeg/OBS/GPU/network are marked (see pyproject markers) and skip
themselves when the dependency is absent. Pure-logic tests run everywhere.
"""

import shutil

import pytest

from dopest_clip import config, project


@pytest.fixture
def projects_root(tmp_path, monkeypatch):
    """Point the project store at an isolated temp dir for the duration of a test."""
    root = tmp_path / "projects"
    root.mkdir()
    monkeypatch.setattr(config, "PROJECTS_ROOT", root)
    return root


@pytest.fixture
def synthetic_transcript():
    """A tiny transcript with clean silence boundaries around each word.

    Words at 1s intervals, each 0.5s long, with a 0.5s silence after each — so every
    word boundary is a clean cut point. Indices are contiguous 0..5.
    """
    words = []
    # leading silence so word 0 also has a clean boundary before it
    silences = [{"start": 0.0, "end": 0.5, "dur": 0.5}]
    t = 0.5
    for i, w in enumerate(["hello", "world", "this", "is", "a", "test"]):
        start = round(t, 3)
        end = round(t + 0.5, 3)
        words.append({"i": i, "w": w, "start": start, "end": end})
        silences.append({"start": end, "end": round(end + 0.5, 3), "dur": 0.5})
        t = end + 0.5
    return {"language": "en", "words": words, "silences": silences}


def have_ffmpeg() -> bool:
    return shutil.which(config.FFMPEG) is not None and shutil.which(config.FFPROBE) is not None


needs_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
