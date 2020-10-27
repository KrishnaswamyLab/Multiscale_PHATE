import tasklogger
import time
import numpy as np
import logging
import platform
import sys


def test_tasks():
    logger = tasklogger.TaskLogger("test_tasks")
    logger.start_task("test")
    assert time.time() - logger.tasks["test"] < 0.01
    time.sleep(logger.min_runtime)
    runtime = logger.complete_task("test")
    assert runtime is not None
    assert runtime >= logger.min_runtime
    assert "test" not in logger.tasks
    logger.complete_task("another test")


def test_log():
    logger = tasklogger.TaskLogger("test_log")
    logger.debug("debug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")


def test_level():
    logger = tasklogger.TaskLogger("test_level")
    logger.set_level(2)
    assert logger.level == logging.DEBUG
    assert logger.logger.level == logging.DEBUG
    logger.set_level(1)
    assert logger.level == logging.INFO
    assert logger.logger.level == logging.INFO
    logger.set_level(0)
    assert logger.level == logging.WARNING
    assert logger.logger.level == logging.WARNING


def test_indent():
    logger = tasklogger.TaskLogger("test_indent")
    logger.set_indent(0)
    assert logger.indent == 0


def test_cpu_timer():
    logger = tasklogger.TaskLogger("test_cpu_timer")
    logger.set_timer("cpu")
    logger.start_task("test")
    time.sleep(logger.min_runtime * 10)
    runtime = logger.complete_task("test")
    assert runtime is not None
    assert runtime < logger.min_runtime * 10


def test_custom_timer():
    logger = tasklogger.TaskLogger("test_custom_timer")
    logger.set_timer(lambda: 1000 * time.time())
    logger.start_task("test")
    time.sleep(logger.min_runtime)
    runtime = logger.complete_task("test")
    assert runtime is not None
    assert runtime >= 1000 * logger.min_runtime


def test_bad_timer():
    logger = tasklogger.TaskLogger("test_bad_timer")
    np.testing.assert_raises(ValueError, logger.set_timer, "bad")


def test_duplicate():
    logger = tasklogger.TaskLogger("test_duplicate")
    np.testing.assert_raises(RuntimeError, tasklogger.TaskLogger, "test_duplicate")
    logger2 = tasklogger.TaskLogger("test_no_duplicate")
    assert logger.logger is not logger2.logger


def test_context():
    logger = tasklogger.TaskLogger("test_context")
    with logger.task("test"):
        assert "test" in logger.tasks
    assert "test" not in logger.tasks
