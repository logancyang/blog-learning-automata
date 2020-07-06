---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Review] Part V: Packages"
comments: true
---

## Packages

This part is about

- how to organize your code into a package structure
- the installation of third party packages
- preparing to give your own code away to others.

Since the landscape of Python packaging tools is evolving, the main focus of this section is on some **general code organization principles** that will prove useful no matter what tools you later use to give code away or manage dependencies.

If writing a larger program, you don’t really want to organize it as a large of collection of standalone files at the top level. Here is how you can organize the files in hierarchy.

### Modules

Any Python source file is a module.

```py
# foo.py
def grok(a):
    ...
def spam(b):
    ...
```

An `import` statement loads and **executes** a module.

```py
# program.py
import foo

a = foo.grok(2)
b = foo.spam('Hello')
...
```

### Packages vs Modules

For larger collections of code, it is common to organize modules into a package.

```
# From this
pcost.py
report.py
fileparse.py

# To this
porty/
    __init__.py
    pcost.py
    report.py
    fileparse.py
```

You pick a name and make a top-level directory. `porty` in the example above (picking this name is the most important first step).

Add an `__init__.py` file to the directory. It may be empty.

Put your source files into the directory.

### Using a package

A package serves as a **namespace** for imports.

This means that there are now multilevel imports.

```py
import porty.report
port = porty.report.read_portfolio('port.csv')

# Or

from porty import report
port = report.read_portfolio('portfolio.csv')

from porty.report import read_portfolio
port = read_portfolio('portfolio.csv')
```

There are two main problems with this approach.

- imports between files in the same package break.
- main scripts placed inside the package break.

#### Problem: imports

Imports **between files in the same package** must now **include the package name in the import**.

Remember the structure.

```
porty/
    __init__.py
    pcost.py
    report.py
    fileparse.py
```

Modified import example:

```py
# report.py
from porty import fileparse

def read_portfolio(filename):
    return fileparse.parse_csv(...)
```

**These imports are absolute**, not relative.

```py
# report.py
import fileparse    # BREAKS. fileparse not found
```

#### Use relative imports inside a package

Instead of directly using the package name, you can use `.` to refer to the current package.

```py
# report.py
from . import fileparse

def read_portfolio(filename):
    return fileparse.parse_csv(...)
```

Using `from . import modname` **makes it easy to rename the package**.

#### Problem: Main Scripts

Running a package submodule as a main script breaks.

```bash
bash $ python porty/pcost.py # BREAKS
```

Reason: You are running Python on a single file and Python doesn’t see the rest of the package structure correctly (`sys.path` is wrong).

All imports break.

Solution: run your program in a different way, **using the `-m` option**.

```bash
bash $ python -m porty.pcost # WORKS
```

#### `__init__.py` files

The primary purpose of these files is to stitch modules together.

Example: consolidating functions

```py
# porty/__init__.py
from .pcost import portfolio_cost
from .report import portfolio_report
```

**This makes names appear at the *top-level* when importing.**

```py
from porty import portfolio_cost
portfolio_cost('portfolio.csv')
```

instead of using the multilevel imports.

```py
from porty import pcost
pcost.portfolio_cost('portfolio.csv')
```

#### Another solution for scripts

As mentioned, you now need to use -m package.module to run scripts within your package.

```py
bash % python3 -m porty.pcost portfolio.csv
```

There is another alternative: Write a new top-level script.

```py
#!/usr/bin/env python3
# pcost.py
import porty.pcost
import sys
porty.pcost.main(sys.argv)
```

This script lives **outside** the package. For example, looking at the directory structure:

```
pcost.py       # top-level-script
porty/         # package directory
    __init__.py
    pcost.py
    ...
```

## Application structure

Code organization and file structure is key to the maintainability of an application.

There is no “one-size fits all” approach for Python.

However, **one structure that works for a lot of problems is something like this.**

```
porty-app/
  README.txt
  script.py         # SCRIPT
  porty/
    # LIBRARY CODE
    __init__.py
    pcost.py
    report.py
    fileparse.py
```

The top-level `porty-app` is a container for everything else –- documentation, top-level scripts, examples, etc.

Again, top-level scripts (if any) need to exist outside the code package. One level up.

## Third-party packages

Python has a large library of **built-in modules**.

There are even more **third party modules**. Check them in the **[Python Package Index](https://pypi.org/)** or **PyPi**. Or just do a Google search for a specific topic.

How to handle third-party dependencies is an ever-evolving topic with Python. This section merely covers the basics to help you wrap your brain around how it works.

### The Module Search Path

**`sys.path` is a directory that contains the list of all directories checked by the import statement**. Look at it:

```py
>>> import sys
>>> sys.path
... look at the result ...
```

If you import something and it’s not located in one of those directories, you will get an ImportError exception.

### Standard Library Modules

Modules from Python’s standard library usually come from a location such as `/usr/local/lib/python3.6`. You can find out for certain by trying a short test:

```py
>>> import re
>>> re
<module 're' from '/usr/local/lib/python3.6/re.py'>
```

**Simply looking at a module in the REPL is a good debugging tip to know about. It will show you the location of the file**.

### Third-party modules

Third party modules are usually located in a dedicated `site-packages` directory. You’ll see it if you perform the same steps as above:

```py
>>> import numpy
>>> numpy
<module 'numpy' from '/usr/local/lib/python3.6/site-packages/numpy/__init__.py'>
```

Again, looking at a module is a good debugging tip if you’re trying to figure out why something related to `import` isn’t working as expected.

### Installing modules

The most common technique for installing a third-party module is to use `pip`. For example:

```bash
bash % python3 -m pip install packagename
```

This command will download the package and install it in the `site-packages` directory.

#### Problems

- You may be using an installation of Python that you don’t directly control.
  - A corporate approved installation
  - You’re using the Python version that comes with the OS.
- You might not have permission to install global packages in the computer.
- There might be other dependencies.

Use virtual environment!

### Virtual environments

A common solution to package installation issues is to create a so-called “virtual environment” for yourself. Naturally, there is no “one way” to do this–in fact, there are several competing tools and techniques. However, if you are using a standard Python installation, you can try typing this:

```bash
bash % python -m venv mypython
```

After a few moments of waiting, you will have a new directory `mypython` that’s your own little Python install. Within that directory you’ll find a `bin/` directory (Unix) or a `Scripts/` directory (Windows). If you run the `activate` script found there, it will “activate” this version of Python, **making it the default python command for the shell**. For example:

```bash
bash % source mypython/bin/activate
(mypython) bash %
```

From here, you can now start installing Python packages for yourself. For example:

```bash
(mypython) bash % python -m pip install pandas
```

For the purposes of experimenting and trying out different packages, a virtual environment will usually work fine. **If, on the other hand, you’re creating an application and it has specific package dependencies, that is a slightly different problem.**

### Handling Third-Party Dependencies in Your Application

If you have written an application and it has specific third-party dependencies, one challange concerns the creation and preservation of the environment that includes your code and the dependencies.

The current (2020) recommendation is to use [Poetry](https://python-poetry.org/).

Refer to the [Python Packaging User Guide](https://packaging.python.org/) for the most up-to-date guide.

## Distribution

At some point you might want to give your code to someone else, possibly just a co-worker. This section gives the most basic technique of doing that. For more detailed information, consult the [Python Packaging User Guide](https://packaging.python.org/).

### Creating a `setup.py` file

Add a `setup.py` file to the top-level of your project directory.

```py
# setup.py
import setuptools

setuptools.setup(
    name="porty",
    version="0.0.1",
    author="Your Name",
    author_email="you@example.com",
    description="Practical Python Code",
    packages=setuptools.find_packages(),
)
```
### Creating `MANIFEST.in`

If there are additional files associated with your project, specify them with a `MANIFEST.in` file. For example:

```bash
# MANIFEST.in
include *.csv
```

Put the `MANIFEST.in` file in the same directory as `setup.py`.

### Creating a source distribution

To create a distribution of your code, use the `setup.py` file. For example:

```bash
bash % python setup.py sdist
```

This will create a `.tar.gz` or `.zip` file in the directory `dist/`. That file is something that you can now give away to others.

### Installing your code

Others can install your Python code using `pip` in the same way that they do for other packages. They simply need to supply the file created in the previous step. For example:

```bash
bash % python -m pip install porty-0.0.1.tar.gz
```

### Comment

The steps above describe the absolute most minimal basics of creating a package of Python code that you can give to another person. In reality, it can be much more complicated depending on third-party dependencies, whether or not your application includes foreign code (i.e., C/C++), and so forth. We’ve only taken a tiny first step.

Refer to the [official guide](https://packaging.python.org/tutorials/packaging-projects/) to see how to upload your package to PyPi.

For a deeper discussion and selection of virtual environment, application dependency management tools, check another [post](http://blog.logancyang.com/note/python/2020/06/17/pip-vs-conda.html) dedicated to this topic.


## Reference

- <https://dabeaz-course.github.io/practical-python/Notes>
- <https://packaging.python.org/tutorials/packaging-projects/>
- <http://blog.logancyang.com/note/python/2020/06/17/pip-vs-conda.html>
