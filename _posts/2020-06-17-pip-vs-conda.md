---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Essentials] Digging into Package and Environment Management to Find the Best Approach"
comments: true
---

This post is for clarifying several points about Python package and environment management.

- `pip` vs. `conda`?
- `virtualenv` vs. `conda` env?
- How to choose the package and environment tools for Python development?

## `pip` vs. `conda`

### Difference 1

`pip` is a ***Python* package manager**.

`conda` is a **language-agnostic** cross-platform ***environment* manager**.

Package management and environment management are two different things. So `pip` and `conda` are different things and do not compete with each other.

Pip installs Python software packaged as **wheels or source distributions**. Source distributions may require that the system have compatible compilers, and possibly libraries, installed before invoking pip to succeed.

Conda packages are binaries. There is never a need to have compilers available to install them. Additionally conda packages are not limited to Python software. They may also contain C or C++ libraries, R packages or any other software.

Before using pip, a Python interpreter must be installed via a system package manager or by downloading and running an installer. Conda on the other hand can install Python packages as well as the Python interpreter directly.

### Difference 2

`conda` has the ability to create isolated environments, `pip` needs `virtualenv` or `venv` to create a virtual environment.

(`poetry` is a tool that wraps `pip` and `virtualenv` together and is recommended for application development. Will discuss in later sections.)

### Difference 3

When installing packages, pip installs dependencies in a recursive, serial loop. No effort is made to ensure that the dependencies of all packages are fulfilled simultaneously. This can lead to environments that are broken in subtle ways, if packages installed earlier in the order have incompatible dependency versions relative to packages installed later in the order. In contrast, conda uses a satisfiability (SAT) solver to verify that all requirements of all packages installed in an environment are met. This check can take extra time but helps prevent the creation of broken environments. As long as package metadata about dependencies is correct, conda will predictably produce working environments.

<img src="{{ site.baseurl }}/images/misc/pipvconda.png" alt="pipvconda" align="middle"/>

Now we know the difference between `pip` and `conda`, but choosing which to use also depends on the way they work with virtual environments. I will discuss that in later sections. Now let's understand some basics how Python installs pacakges, and check the resulting package locations after `pip install` and `conda install`.

## Where does Python install packages (mac system Python, DON'T DO THIS)

By default, without a virtual environment, all `python` and `pip` commands will use the default executables, usually your **system Python install**.

***It is strongly recommended to keep your system Python clean of unnecessary site packages by using virtual environments***.

Otherwise, over time, you will add lots of things to your system packages and things might conflict and cause problems. Using an isolated environment for each project ensures easy reproducability and reduced conflict.

> DO NOT install into your global Python interpreter! ALWAYS use an environment when developing locally!

To see what directories are being used to search for packages, invoke the `site` module directly by running `python -m site` outside virtual environments:

```bash
$ python -m site

sys.path = [
    '<current_dir>',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python27.zip',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-darwin',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/plat-mac/lib-scriptpackages',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-tk',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-old',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload',
    '/Library/Python/2.7/site-packages',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python',
    '/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/PyObjC',
]
USER_BASE: '/Users/<username>/Library/Python/2.7' (doesn't exist)
USER_SITE: '/Users/<username>/Library/Python/2.7/lib/python/site-packages' (doesn't exist)
ENABLE_USER_SITE: True
```

The Python at `/System/Library/Frameworks/...` is MacOS's pre-installed Python.

Run `pip show <package_name>` to see where it's installed. For system Python, it's something like:

```bash
$ pip show setuptools

Name: setuptools
Version: 41.0.1
Summary: Easily download, build, install, upgrade, and uninstall Python packages
Home-page: https://github.com/pypa/setuptools
Author: Python Packaging Authority
Author-email: distutils-sig@python.org
License: UNKNOWN
Location: /System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python
Requires:
Required-by:
```

You can also run `pip list -v` to check install locations.

## `pip install` vs. `conda install`: where do they install the packages

If you run `pip install` in a virtual environment, by default it will install into

```
<project_root>/<virtualenv_name>/lib/<python_ver>/site-packages/
```

For `conda`, it always installs packages into a conda environment. Without an activated conda environment, `conda install` will install into the `base` conda environment. With a conda environment, e.g. `testenv` activated, *by default* it installs to

```
/home/<user>/anaconda3/envs/testenv/
```

(**`conda` envs are located in a centralized path by default**. Later I will mention the pros and cons of this approach.)

We can use `conda list` to check the path of the current environment and the packages installed in it.

Later I will compare the two and choose my preferred way to manage packages and environments.

## `virtualenv` vs conda env

With `pip` vs. `conda` out of the way, a more valid comparison is `virtualenv` vs. conda environments. When to use which? To answer that, we need to really understand what virtual environment is and how they achieve isolation.

### Always use a virtual environment!

A virtual environment is **a directory that contains its own installation of Python and its own set of libraries (site packages)**. It solves problems such as conflicting requirements for two applications by having an isolated environment for each application.

### How does `virtualenv` achieve isolation

Notice the difference between the first path in `$PATH` before and after the activation:

```bash
$ echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:

$ source env/bin/activate
(env) $ echo $PATH
/Users/<username>/<project>/venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:
```

This raises the following questions:

- What’s the difference between these two executables anyway?
- How is the virtual environment’s Python executable able to use something other than the system’s site-packages?

This can be explained by how Python starts up and where it is located on the system. There actually isn’t any difference between these two Python executables. **It’s their directory locations that matter**.

When Python is starting up, it looks at the **path of its binary**. In a virtual environment, it is actually just a copy of, or symlink to, your system’s Python binary. It then sets the location of `sys.prefix` and `sys.exec_prefix` based on this location, omitting the bin portion of the path.

The path located in `sys.prefix` is then used for locating the `site-packages` directory by searching the relative path `lib/pythonX.X/site-packages/`, where X.X is the version of Python you’re using.

In our example, the binary is located at

```
/Users/<username>/<project>/venv/bin
```

which means `sys.prefix` would be

```
/Users/<username>/<project>/venv/
```

and therefore the site-packages directory used would be

```
/Users/<username>/<project>/venv/lib/pythonX.X/site-packages
```

Finally, this path is stored in the `sys.path` array, which contains all of the locations where a package can reside.

### Considerations for picking `virtualenv` or `conda` env

`conda` env *can* replace `virtualenv`, but there are some differences. There is a [great answer](https://stackoverflow.com/a/59755292/2280673) from stackoverflow, I quote and arrange into the points below.

#### Where do the envs live: central vs. per-project

By default `conda` prefers to manage a list of environments for you in a central location, i.e.

```
/home/<user>/anaconda3/envs/
```

whereas `virtualenv` makes a folder in the current directory.

- `conda` (centralized) makes sense if you are e.g. doing machine learning and just have a couple of broad environments that you use across many projects and want to jump into them from anywhere.
- `virtualenv` (per project folder) makes sense if you are doing projects that have completely different sets of lib requirements that really belong more to the project itself.

#### Environment size (favors `virtualenv`)

The empty environment that Conda creates is about 122MB whereas the virtualenv's is about 12MB, so that's another reason you may prefer not to scatter Conda environments around everywhere.

#### Prefix for shell display (favors `virtualenv`)

Finally, another superficial indication that `conda` prefers its centralized envs is that (again, by default) if you do create a `conda` env in your own project folder and activate it the name prefix that appears in your shell is the (way too long) absolute path to the folder. You can fix that by giving it a name, but `virtualenv` does the right thing by default.

### Weird question of mine: what if `virtualenv` and conda env are both activated?

I did this experiment by activating one before the other and check `which python`, and guess what, the result is that the active environment is whichever environment **activated last**!

### Some tips

- Avoid using `pip` by itself. Using `python -m pip` will always guarantee you are using the pip associated with that specific python being called, instead of potentially calling a pip associated with a different python.
- Sometimes there is a pip3 to go with python3 to differentiate python 2 executables python/pip.
- Sometimes Linux distributions require you to install virtualenv as a separate package. For example `sudo apt install python3-virtualenv`.
- You should never copy or move around virtual environments. Always create new ones.
Ignore the virtual environment directories from repositories. For example, `.gitignore` them.

## pyenv

`pyenv` is a mature tool for installing and managing multiple Python versions on macOS. You can install it with `brew`. If you’re using Windows, you can use `pyenv-win`. After you’ve got `pyenv` installed, you can install multiple versions of Python into your Python environment with a few short commands:

```bash
$ pyenv versions
* system
$ python --version
Python 2.7.10
$ pyenv install 3.7.3  # This may take some time
$ pyenv versions
* system
  3.7.3
```

You can manage which Python you’d like to use in your current session, globally, or on a per-project basis as well. pyenv will make the python command point to whichever Python you specify. Note that none of these overrides the default system Python for other applications, so you’re safe to use them however they work best for you within your Python environment:

```bash
$ pyenv global 3.7.3
$ pyenv versions
  system
* 3.7.3 (set by /Users/<username>/.pyenv/version)

$ pyenv local 3.7.3
$ pyenv versions
  system
* 3.7.3 (set by /Users/<username>/myproj/.python-version)

$ pyenv shell 3.7.3
$ pyenv versions
  system
* 3.7.3 (set by PYENV_VERSION environment variable)

$ python --version
Python 3.7.3
```

**There is something called `pyenv virtualenv`.** Install it with `brew install pyenv-virtualenv` and use it with `pyenv virtualenv <python version> <project name>`.

```bash
// Create virtual environment
$ pyenv virtualenv 3.7.3 my-env

// Activate virtual environment
$ pyenv activate my-env

// Exit virtual environment
(my-env)$ pyenv deactivate

$ pyenv virtualenv 3.7.3 proj1
$ pyenv virtualenv 3.7.3 proj2
$ cd /Users/<username>/proj1
$ pyenv local proj1
(proj1)$ cd ../proj2
$ pyenv local proj2
(proj2)$ pyenv versions
  system
  3.7.3
  3.7.3/envs/proj1
  3.7.3/envs/proj2
  proj1
* proj2 (set by /Users/<username>/proj2/.python-version)
```

**It can set the default environment for a directory! This way you don't have to remember which environment to use.** But the caveat is that it has conda-like centralized envs. You can find them by the command `pyenv virtualenvs`.

For more details, check out <https://realpython.com/intro-to-pyenv/> and <https://github.com/pyenv/pyenv-virtualenv>.

## My setup 2020

Tools:
- `pyenv`, `virtualenv` for **library** development.
- `poetry` for **application** development. `poetry` wraps `pip` and `virtualenv` to provide a unified method for working with these environments.

I use `virtualenv` instead of `pyenv-virtualenv` to explicitly have a folder for the virtual environment in the project directory because it is the clearest way for me to manage virtual environments for different projects.

`conda` can be used for **one-off experiments** with broad machine learning environments.

For a comparison of library vs. application development and more Python packaging considerations, check out this great article: [The Packaging Gradient](https://sedimental.org/the_packaging_gradient.html).

> Application packaging must not be confused with library packaging. Python is for both, but pip is for libraries.

### Quick intro to `poetry`

Use `poetry` instead of `pipenv` for Python application development. `pipenv` is [not actively issuing updates as of 2020](https://packaging.python.org/tutorials/managing-dependencies/#recommendation-caveats-as-of-april-2020).

`poetry` addresses additional facets of package management, including creating and publishing your own packages. After installing `poetry`, you can use it to create a new project:

```bash
$ poetry new myproj
Created package myproj in myproj
$ ls myproj/
README.rst    myproj    pyproject.toml    tests
```

Similarly to how pipenv creates the Pipfile, poetry creates a pyproject.toml file. This recent standard contains metadata about the project as well as dependency versions:

```
# Config file
[tool.poetry]
name = "myproj"
version = "0.1.0"
description = ""
authors = ["Logan Yang <logancyang@gmail.com>"]

[tool.poetry.dependencies]
python = "^3.7"

[tool.poetry.dev-dependencies]
pytest = "^3.0"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

You can install packages with poetry add (or as development dependencies with poetry add --dev):

```bash
$ poetry add requests
Using version ^2.22 for requests

Updating dependencies
Resolving dependencies... (0.2s)

Writing lock file


Package operations: 5 installs, 0 updates, 0 removals

  - Installing certifi (2019.6.16)
  - Installing chardet (3.0.4)
  - Installing idna (2.8)
  - Installing urllib3 (1.25.3)
  - Installing requests (2.22.0)
```

`poetry` also maintains a lock file, and it has a benefit over `pipenv` because it keeps track of which packages are subdependencies. As a result, you can uninstall `requests` and its dependencies with `poetry remove requests`.

## References

- <https://www.anaconda.com/blog/understanding-conda-and-pip>
- <https://jakevdp.github.io/blog/2016/08/25/conda-myths-and-misconceptions/>
- <https://stackoverflow.com/questions/54834579/specific-reasons-to-favor-pip-vs-conda-when-installing-python-packages>
- <https://stackoverflow.com/a/59755292/2280673>
- <https://realpython.com/effective-python-environment/>
- <https://realpython.com/python-virtual-environments-a-primer/>
- <https://packaging.python.org/tutorials/managing-dependencies/>
- <https://sedimental.org/the_packaging_gradient.html>
- <https://youtu.be/3J02sec99RM>
- <https://youtu.be/o1Vue9CWRxU>
