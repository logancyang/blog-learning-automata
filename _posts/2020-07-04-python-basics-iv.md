---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Review] Part IV: Testing, Logging and Debugging"
comments: true
---

## Testing

The dynamic nature of Python makes testing critically important to most applications. There is no compiler to find your bugs. The only way to find bugs is to run the code and make sure you try out all of its features.

### Assertions

The `assert` statement is an internal check for the program. If an expression is not true, it raises a `AssertionError` exception.

`assert` statement syntax:

```py
assert <expression> [, 'Diagnostic message']
```

For example,

```py
assert isinstance(10, int), 'Expected int'
```

(**BTW, the syntax of nested `[, arg]` in documentation `[, arg1 [, arg2 [...]]]` is the standard Python way of documenting function signatures. Check this [post](https://stackoverflow.com/questions/2120507/why-do-python-function-docs-include-the-comma-after-the-bracket-for-optional-arg) for why.**)

It shouldn’t be used to check the user-input (i.e., data entered on a web form or something). It’s purpose is more for internal checks and invariants (conditions that should always be true).

#### Contract programming

Also known as Design By Contract, liberal use of assertions is an approach for designing software. It prescribes that software designers should define precise interface specifications for the components of the software.

For example, you might put assertions on all inputs of a function.

```py
def add(x, y):
    assert isinstance(x, int), 'Expected int'
    assert isinstance(y, int), 'Expected int'
    return x + y

>>> add(2, 3)
5
>>> add('2', '3')
Traceback (most recent call last):
...
AssertionError: Expected int
```

Checking inputs will immediately catch callers who aren’t using appropriate arguments.

#### Inline tests

Assertions can also be used for simple tests.

```py
def add(x, y):
    return x + y

assert add(2,2) == 4
```

This way you are including the test in the same module as your code. Benefit: **If the code is obviously broken, attempts to import the module will crash.**

This is not recommended for exhaustive testing. It’s more of a basic “smoke test”. Does the function work on any example at all? If not, then something is definitely broken.

### `unittest` module

Suppose you have a code file `simple.py`, and you write `test_simple.py` to test it:

```py
# simple.py

def add(x, y):
    return x + y

# test_simple.py

import simple
import unittest

# Notice that it inherits from unittest.TestCase
class TestAdd(unittest.TestCase):
    def test_simple(self):
        # Test with simple integer arguments
        r = simple.add(2, 2)
        self.assertEqual(r, 5)
    def test_str(self):
        # Test with strings
        r = simple.add('hello', 'world')
        self.assertEqual(r, 'helloworld')
```

**The testing class must inherit from `unittest.TestCase`.**

In the testing class, you can define the testing methods.

*Important*: Each method must start with `test`.

#### Using `unittest`

There are several built in assertions that come with `unittest`. Each of them asserts a different thing.

```py
# Assert that expr is True
self.assertTrue(expr)

# Assert that x == y
self.assertEqual(x,y)

# Assert that x != y
self.assertNotEqual(x,y)

# Assert that x is near y
self.assertAlmostEqual(x,y,places)

# Assert that callable(arg1,arg2,...) raises exc
self.assertRaises(exc, callable, arg1, arg2, ...)
```

This is not an exhaustive list. There are other assertions in the module.

#### Running `unittest`

To run the tests, turn the code into a script.

```py
# test_simple.py

...

if __name__ == '__main__':
    unittest.main()
```

Then run Python on the test file.

```
bash % python3 test_simple.py
F.
========================================================
FAIL: test_simple (__main__.TestAdd)
--------------------------------------------------------
Traceback (most recent call last):
  File "testsimple.py", line 8, in test_simple
    self.assertEqual(r, 5)
AssertionError: 4 != 5
--------------------------------------------------------
Ran 2 tests in 0.000s
FAILED (failures=1)
```

When you run a test that checks the type of something, you need to check that an exception is raised:

```py
class TestStock(unittest.TestCase):
    ...
    def test_bad_shares(self):
        s = stock.Stock('GOOG', 100, 490.1)
        with self.assertRaises(TypeError):
            s.shares = '100'
```

Effective unit testing is an art and it can grow to be quite complicated for large applications.

The `unittest` module has a huge number of options related to test runners, collection of results and other aspects of testing. Consult the documentation for details.

### Third-party test tools: `pytest`

The built-in unittest module has the advantage of being available everywhere–it’s part of Python. However, many programmers also find it to be quite verbose. A popular alternative is [pytest](https://docs.pytest.org/en/latest/). With pytest, your testing file simplifies to something like the following:

```py
# test_simple.py
import simple

def test_simple():
    assert simple.add(2,2) == 4

def test_str():
    assert simple.add('hello','world') == 'helloworld'
```

To run the tests, you simply type a command such as `python -m pytest`. It will discover all of the tests and run them.

There’s a lot more to `pytest` than this example, but it’s usually pretty easy to get started should you decide to try it out.

## Logging

The `logging` module is a standard library module for recording diagnostic information. It’s also a very large module with a lot of sophisticated functionality. We will show a simple example to illustrate its usefulness.

Suppose we have:

```py
# fileparse.py
def parse(f, types=None, names=None, delimiter=None):
    records = []
    for line in f:
        line = line.strip()
        if not line: continue
        try:
            records.append(split(line,types,names,delimiter))
        except ValueError as e:
            print("Couldn't parse :", line)
            print("Reason :", e)
    return records
```

We shouldn't print the message when there's an exception. Answer from [stackoverflow](https://stackoverflow.com/questions/6918493/in-python-why-use-logging-instead-of-print):

> The logging package has a lot of useful features:

> - Easy to see where and when (even what line no.) a logging call is being made from.
> - You can log to files, sockets, pretty much anything, all at the same time.
> - You can differentiate your logging based on severity.

> Print doesn't have any of these.

>Also, if your project is meant to be imported by other python tools, it's bad practice for your package to print things to stdout, since the user likely won't know where the print messages are coming from. With logging, users of your package can choose whether or not they want to propogate logging messages from your tool or not.

To use `logging` here:

```py
# fileparse.py
import logging
# CREATE THE LOGGER OBJECT
log = logging.getLogger(__name__)

def parse(f,types=None,names=None,delimiter=None):
    ...
    try:
        records.append(split(line,types,names,delimiter))
    except ValueError as e:
        log.warning("Couldn't parse : %s", line)
        log.debug("Reason : %s", e)
```

The code is modified to issue warning messages via a `Logger` object. The one created with `logging.getLogger(__name__)`.

### Logging basics

**Create a logger object:**

```py
logger = logging.getLogger(name)   # name is a string
```

Issuing log messages.

```py
logger.critical(message [, args])
logger.error(message [, args])
logger.warning(message [, args])
logger.info(message [, args])
logger.debug(message [, args])
```

Each method represents a **different level of severity**.

All of them create a formatted log message. `args` is used with the `%` operator to create the message.

```py
logmsg = message % args # Written to the log
```

### Logging configuration

Typically, this is a one-time configuration at program startup. The configuration is separate from the code that makes the logging calls.

```py
# main.py

...

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        filename  = 'app.log',      # Log output file
        level     = logging.INFO,   # Output level
    )
```

Logging is highly configurable. You can adjust every aspect of it: output files, levels, message formats, etc. However, the code that uses logging doesn’t have to worry about that.

```py
# Set level to DEBUG
>>> logging.getLogger('fileparse').level = logging.DEBUG
>>> a = report.read_portfolio('Data/missing.csv')
WARNING:fileparse:Row 4: Bad row: ['MSFT', '', '51.23']
DEBUG:fileparse:Row 4: Reason: invalid literal for int() with base 10: ''
WARNING:fileparse:Row 7: Bad row: ['IBM', '', '70.44']
DEBUG:fileparse:Row 7: Reason: invalid literal for int() with base 10: ''
>>>
# Turn it off, set level to CRITICAL
>>> logging.getLogger('fileparse').level=logging.CRITICAL
>>> a = report.read_portfolio('Data/missing.csv')
>>>
```

### Adding logging to a program

**To add logging to an application, you need to have some mechanism to initialize the `logging` module in the *main module*. One way to do this is to include some setup code that looks like this:**

```py
# This file sets up basic configuration of the logging module.
# Change settings here to adjust logging output as needed.
import logging
logging.basicConfig(
    # Name of the log file (omit to use stderr)
    filename = 'app.log',
    # File mode (use 'a' to append)
    filemode = 'w',
    # Logging level (DEBUG, INFO, WARNING, ERROR, or CRITICAL)
    level    = logging.WARNING,
)
```

You’d need to put this someplace in the startup steps of your program.

## Debugging

So, your program has crashed... Now what? Read the tracebacks!

```bash
bash % python3 blah.py
Traceback (most recent call last):
  File "blah.py", line 13, in ?
    foo()
  File "blah.py", line 10, in foo
    bar()
  File "blah.py", line 7, in bar
    spam()
  File "blah.py", 4, in spam
    line x.append(3)
# Cause of the crash
AttributeError: 'int' object has no attribute 'append'
```

PRO TIP: Paste the whole traceback to Google!

You can use `-i` to keep Python alive to Python REPL when executing in the shell:

```bash
bash % python3 -i blah.py
Traceback (most recent call last):
  File "blah.py", line 13, in ?
    foo()
  File "blah.py", line 10, in foo
    bar()
  File "blah.py", line 7, in bar
    spam()
  File "blah.py", 4, in spam
    line x.append(3)
AttributeError: 'int' object has no attribute 'append'
>>>
```

It preserves the interpreter state. That means that you can go poking around after the crash. Checking variable values and other state.

### Debugging with `print()`

Tip: use `repr()`. It shows details.

```py
def spam(x):
    print('DEBUG:', repr(x))
    ...

>>> from decimal import Decimal
>>> x = Decimal('3.4')
# NO `repr`
>>> print(x)
3.4
# WITH `repr`
>>> print(repr(x))
Decimal('3.4')
>>>
```

### Debugging with the Python debugger `pdb`

You can manually launch the debugger inside a program.

```py
def some_function():
    ...
    breakpoint()      # Enter the debugger (Python 3.7+)
    ...
```

This starts the debugger at the `breakpoint()` call.

In earlier Python versions, you did this,

```py
import pdb
...
pdb.set_trace()       # Instead of `breakpoint()`
...
```

You can also run the entire program under debugger:

```bash
bash % python3 -m pdb someprogram.py
```

It will automatically enter the debugger before the first statement. Allowing you to set breakpoints and change the configuration.

Common debugger commands:

```
(Pdb) help            # Get help
(Pdb) w(here)         # Print stack trace
(Pdb) d(own)          # Move down one stack level
(Pdb) u(p)            # Move up one stack level
(Pdb) b(reak) loc     # Set a breakpoint
(Pdb) s(tep)          # Execute one instruction
(Pdb) c(ontinue)      # Continue execution
(Pdb) l(ist)          # List source code
(Pdb) a(rgs)          # Print args of current function
(Pdb) !statement      # Execute statement
```

Example for setting breakpoints:

```
(Pdb) b 45            # Line 45 in current file
(Pdb) b file.py:45    # Line 34 in file.py
(Pdb) b foo           # Function foo() in current file
(Pdb) b module.foo    # Function foo() in a module
```

### Develop using Jupyter Notebook and `nbdev`

Using Jupyter Notebook and `nbdev` by fast.ai is good for efficient debugging. Jupyter Notebook is a great tool for experimenting with code.

### Debugging in VSCode

You can create a `py` file and use `# %%` to specify a cell which behaves like a Jupyter Notebook cell. Click at the beginning of any line in the source code to set a breakpoint, and run the cell with `Debug cell`.

<img src="{{ site.baseurl }}/images/misc/vscode-debug.png" alt="" align="middle"/>

## Reference

- <https://dabeaz-course.github.io/practical-python/Notes>
- <https://stackoverflow.com/questions/2120507/why-do-python-function-docs-include-the-comma-after-the-bracket-for-optional-arg>
- <https://stackoverflow.com/questions/6918493/in-python-why-use-logging-instead-of-print>
