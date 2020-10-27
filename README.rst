==========
tasklogger
==========

.. image:: https://img.shields.io/pypi/v/tasklogger.svg
    :target: https://pypi.org/project/tasklogger/
    :alt: Latest PyPi version
.. image:: https://anaconda.org/conda-forge/tasklogger/badges/version.svg
    :target: https://anaconda.org/conda-forge/tasklogger/
    :alt: Latest Conda version
.. image:: https://api.travis-ci.com/scottgigante/tasklogger.svg?branch=master
    :target: https://travis-ci.com/scottgigante/tasklogger
    :alt: Travis CI Build
.. image:: https://ci.appveyor.com/api/projects/status/qi79tqay73uslr0i/branch/master?svg=true
    :target: https://ci.appveyor.com/project/scottgigante/tasklogger
    :alt: Appveyor Build
.. image:: https://coveralls.io/repos/github/scottgigante/tasklogger/badge.svg?branch=master
    :target: https://coveralls.io/github/scottgigante/tasklogger?branch=master
    :alt: Coverage Status
.. image:: https://img.shields.io/twitter/follow/scottgigante.svg?style=social&label=Follow
    :target: https://twitter.com/scottgigante
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/scottgigante/tasklogger.svg?style=social&label=Stars
    :target: https://github.com/scottgigante/tasklogger/
    :alt: GitHub stars
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

An extension to the core python logging library for logging the beginning and completion of tasks and subtasks.

Installation
------------

tasklogger is available on `pip`. Install by running the following in a terminal::

    pip install --user tasklogger

Alternatively, tasklogger can be installed using `Conda <https://conda.io/docs/>`_ (most easily obtained via the `Miniconda Python distribution <https://conda.io/miniconda.html>`_)::

    conda install -c conda-forge tasklogger

Usage examples
--------------

Receive timed updates mid-computation using ``tasklogger.log_start`` and ``tasklogger.log_complete``::

    >>> import tasklogger
    >>> import time
    >>> tasklogger.log_start("Supertask")
    Calculating Supertask...
    >>> time.sleep(1)
    >>> tasklogger.log_start("Subtask")
      Calculating Subtask...
    >>> time.sleep(1)
    >>> tasklogger.log_complete("Subtask")
      Calculated Subtask in 1.01 seconds.
    >>> time.sleep(1)
    >>> tasklogger.log_complete("Supertask")
    Calculated Supertask in 3.02 seconds.

Simplify logging syntax with ``tasklogger.log_task``::

    >>> import tasklogger
    >>> import time
    >>> with tasklogger.task("Supertask"):
    ...     time.sleep(1)
    ...     with tasklogger.log_task("Subtask"):
    ...        time.sleep(1)
    ...     time.sleep(1)
    Calculating Supertask...
      Calculating Subtask...
      Calculated Subtask in 1.01 seconds.
    Calculated Supertask in 3.02 seconds.

Log wall time, CPU time, or any other counter function with the class API::

    >>> import tasklogger
    >>> import time
    >>> logger = tasklogger.TaskLogger(name='cpu_logger', timer='cpu', min_runtime=0)
    >>> with logger.task("Supertask"):
    ...     time.sleep(1)
    ...     with logger.task("Subtask"):
    ...        _ = [[(i,j) for j in range(i)] for i in range(1000)]
    ...     time.sleep(1)
    Calculating Supertask...
      Calculating Subtask...
      Calculated Subtask in 0.09 seconds.
    Calculated Supertask in 0.09 seconds.
    >>> logger = tasklogger.TaskLogger(name='nano_logger', timer=time.monotonic_ns)
    >>> with logger.task("Supertask"):
    ...     time.sleep(1)
    ...     with logger.task("Subtask"):
    ...        time.sleep(1)
    ...     time.sleep(1)
    Calculating Supertask...
      Calculating Subtask...
      Calculated Subtask in 1001083511.00 seconds.
    Calculated Supertask in 3003702161.00 seconds.

Use ``tasklogger`` for all your logging needs::

    >>> tasklogger.log_info("Log some stuff that doesn't need timing")
    Log some stuff that doesn't need timing
    >>> tasklogger.log_debug("Log some stuff that normally isn't needed")
    >>> tasklogger.set_level(2)
    Set TaskLogger logging to DEBUG
    >>> tasklogger.log_debug("Log some stuff that normally isn't needed")
    Log some stuff that normally isn't needed
