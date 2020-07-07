---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Essentials] Poetry Quick Start"
comments: true
---

[Poetry](https://python-poetry.org/) is the current best choice for Python application dependency and environment management. It uses two files

- pyproject.toml
- poetry.lock

to manage dependencies instead of

- setup.py
- requirements.txt

## Set up a project

### Prerequisite: `pyenv`

Make sure you properly setup pyenv first. Add the following to `~/.zshrc`,

```zsh
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
```

and do

```zsh
$ pyenv global 3.7.4
```

to set the global default Python version to 3.7.4.

### Install `poetry`

Run the following command to install

```zsh
$ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Now, to make sure `poetry` use the right Python version, add an alias to `~/.zshrc`

```zsh
alias poetry="python3 $HOME/.poetry/bin/poetry"
```

### Create a new project

Run

```zsh
$ poetry new <new_project_name>
```

It will create a directory `<new_project_name>` in the current directory.

Go inside and type `tree` (assuming you have `tree` installed, on MacOS run `brew install tree`) you will see

```
.
├── README.rst
├── <new_project_name>
│   └── __init__.py
├── pyproject.toml
└── tests
    ├── __init__.py
    └── test_<new_project_name>.py
```

with a prefilled `pyproject.toml` file:

```
[tool.poetry]
name = "<new_project_name>"
version = "0.1.0"
description = ""
authors = ["Your name <your email>"]

[tool.poetry.dependencies]
python = "^3.7"

[tool.poetry.dev-dependencies]
pytest = "^5.2"

[build-system]
requires = ["poetry>=0.12"]
build-backend = "poetry.masonry.api"
```

Note that `poetry` gets `"Your name <your email>"` from your `git` setting.

Run

```zsh
$ git config --global user.email <email>
```

to change the git email address if you need to.

Now, we can add a package, for example, `flask`,

```zsh
$ poetry add flask
```

And a virtual environment will be created automatically at `/Users/<username>/Library/Caches/pypoetry/virtualenvs/` by default, and `flask` will be installed there.

Check the lock file for detailed subdependencies.

### Activate that virtual environment in shell

Run the following command to activate the virtual environment.

```zsh
$ poetry shell
```

### Set the interpreter in VSCode

As of June 2020, the Python extension in VSCode doesn't support automatic discovery of poetry environments. Refer to this [issue](https://github.com/microsoft/vscode-python/issues/8372).

The current workaround is to add the path into VSCode setting `python.venvPath`,

```
"python.venvPath": "/Users/<username>/Library/Caches/pypoetry/virtualenvs"
```

Now we can see the virtual environments created by `poetry` in the list in VSCode.

## References

Check the official docs for [poetry commands](https://python-poetry.org/docs/cli/).
