# Copyright 2025-2026 Morphic Inc. Licensed under Apache 2.0.
"""Tests for reshoot.py unified CLI."""
import pytest
import subprocess
import sys
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


class TestCLIHelp:
    def test_main_help(self):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "reshoot.py"), "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "depth" in r.stdout
        assert "visualize" in r.stdout
        assert "render" in r.stdout
        assert "infer" in r.stdout
        assert "validate" in r.stdout

    def test_unknown_command(self):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "reshoot.py"), "nonexistent"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
        assert "Unknown" in r.stdout

    def test_validate_help(self):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "reshoot.py"), "validate", "--help"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0
        assert "condition_pack" in r.stdout


class TestCLIValidate:
    def test_validate_existing_pack(self):
        pack = os.path.join(ROOT, "render_outputs")
        if not os.path.exists(pack):
            pytest.skip("render_outputs not available")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "reshoot.py"), "validate", pack],
            capture_output=True, text=True,
        )
        assert "OK" in r.stdout or r.returncode == 0

    def test_validate_missing_dir(self):
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "reshoot.py"), "validate", "/nonexistent/dir"],
            capture_output=True, text=True,
        )
        assert r.returncode == 1
