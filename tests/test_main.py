"""Tests for the ``main`` module entrypoint helpers."""

from __future__ import annotations

import logging
from unittest.mock import Mock

import main


def test_main_runs_application_success(monkeypatch):
    logger = logging.getLogger("test-logger")
    application = Mock()
    application.run.return_value = 0

    configure = Mock(return_value=logger)
    load = Mock(return_value="settings")
    build = Mock(return_value=application)

    monkeypatch.setattr(main, "configure_logging", configure)
    monkeypatch.setattr(main, "load_settings", load)
    monkeypatch.setattr(main, "build_application", build)

    exit_code = main.main([])

    assert exit_code == 0
    configure.assert_called_once_with(level="INFO")
    load.assert_called_once_with(None)
    build.assert_called_once_with(settings="settings", logger=logger)
    application.run.assert_called_once_with()


def test_main_honours_overrides(monkeypatch):
    logger = logging.getLogger("override-logger")
    application = Mock()
    application.run.return_value = 2

    configure = Mock(return_value=logger)
    load = Mock(return_value="settings")
    build = Mock(return_value=application)

    monkeypatch.setattr(main, "configure_logging", configure)
    monkeypatch.setattr(main, "load_settings", load)
    monkeypatch.setattr(main, "build_application", build)

    exit_code = main.main(["--config", "alt.json", "--log-level", "DEBUG"])

    assert exit_code == 2
    configure.assert_called_once_with(level="DEBUG")
    load.assert_called_once_with("alt.json")
    build.assert_called_once_with(settings="settings", logger=logger)
