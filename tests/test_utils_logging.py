import logging

import pytest

from mgnify_methods.utils.logging import get_logger, Logger


def test_get_logger_writes_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = get_logger("test_logger", log_file=str(log_file))
    logger.info("hello")

    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass

    assert log_file.exists()
    assert "hello" in log_file.read_text()


def test_parse_level_invalid():
    with pytest.raises(ValueError):
        Logger._parse_level("NOT_A_LEVEL")
