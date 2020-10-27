import tasklogger
import time
import logging
import numpy as np
import sys
import platform


def test_get_logger():
    logger = tasklogger.get_tasklogger()
    logger2 = tasklogger.get_tasklogger()
    assert logger is logger2
    logger2 = tasklogger.get_tasklogger("test")
    assert logger is not logger2


def test_tasks():
    logger = tasklogger.log_start("test")
    assert time.time() - logger.tasks["test"] < 0.01
    time.sleep(logger.min_runtime)
    tasklogger.log_complete("test")
    assert "test" not in logger.tasks


def test_log():
    tasklogger.log_debug("debug")
    tasklogger.log_info("info")
    tasklogger.log_warning("warning")
    tasklogger.log_error("error")
    tasklogger.log_critical("critical")


def test_level():
    logger = tasklogger.set_level(2)
    assert logger.level == logging.DEBUG
    assert logger.logger.level == logging.DEBUG


def test_indent():
    logger = tasklogger.set_indent(0)
    assert logger.indent == 0


def test_timer():
    logger = tasklogger.set_timer("wall")
    assert logger.timer == time.time
    timer = lambda: 10
    logger = tasklogger.set_timer(timer)
    assert logger.timer == timer
    if sys.version[0] == "2" and platform.system() == "Windows":
        np.testing.assert_raises(RuntimeError, logger.set_timer, "cpu")
    else:
        logger = tasklogger.set_timer("cpu")
        assert logger.timer == time.process_time


def test_context():
    logger = tasklogger.TaskLogger("test_context_api")
    with tasklogger.log_task("test", logger="test_context_api"):
        assert "test" in logger.tasks
    assert "test" not in logger.tasks
