import tasklogger.stream
import tasklogger.utils
import tasklogger
import numpy as np
import os
import sys


def test_ipynb():
    def monkey_patch():
        return True

    temp = tasklogger.utils.in_ipynb
    tasklogger.utils.in_ipynb = monkey_patch
    logger = tasklogger.TaskLogger("ipynb")
    logger.info("ipynb")
    tasklogger.utils.in_ipynb = temp


def test_oserror():
    def monkey_patch(*args, **kwargs):
        raise OSError("[Errno 9] Bad file descriptor")

    temp = os.write
    os.write = monkey_patch
    logger = tasklogger.TaskLogger("oserror")
    logger.info("oserror")
    os.write = temp


def test_no_stdout():
    temp = sys.stdout
    sys.stdout = None
    logger = tasklogger.TaskLogger("no stdout")
    logger.info("no stdout")
    logger.logger.handlers[0].stream.flush()
    sys.stdout = temp


def test_stderr():
    temp = sys.stdout
    sys.stdout = None
    logger = tasklogger.TaskLogger("stderr", stream="stderr")
    logger.info("stderr")
    sys.stdout = temp


def test_invalid_stream():
    np.testing.assert_raises(
        ValueError, tasklogger.TaskLogger, "invalid stream", stream="invalid"
    )
