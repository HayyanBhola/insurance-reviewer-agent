"""Shared test setup.

Tests must not depend on the switches in your .env (AI_ESCALATION, TYRE_REVIEW, AI_CHECKS):
every test starts from the ORIGINAL defaults, and a test that checks a switch sets it itself.
"""
import pytest


@pytest.fixture(autouse=True)
def _default_switches(monkeypatch):
    monkeypatch.setenv("AI_ESCALATION", "on")
    monkeypatch.setenv("TYRE_REVIEW", "off")
    monkeypatch.setenv("AI_CHECKS", "v2")
