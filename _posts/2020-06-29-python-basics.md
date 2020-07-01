---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Review] Part I: The Basics"
comments: true
---

## String methods

```py
s.endswith(suffix)     # Check if string ends with suffix
s.find(t)              # First occurrence of t in s
s.index(t)             # First occurrence of t in s
s.isalpha()            # Check if characters are alphabetic
s.isdigit()            # Check if characters are numeric
s.islower()            # Check if characters are lower-case
s.isupper()            # Check if characters are upper-case
s.join(slist)          # Join a list of strings using s as delimiter
s.lower()              # Convert to lower case
s.replace(old,new)     # Replace text
s.rfind(t)             # Search for t from end of string
s.rindex(t)            # Search for t from end of string
s.split([delim])       # Split string into list of substrings
s.startswith(prefix)   # Check if string starts with prefix
s.strip()              # Strip leading/trailing space
s.upper()              # Convert to upper case
```

Strings are immutable. **All operations and methods that manipulate string data, always create new strings.**

### Byte strings

A string of 8-bit bytes, commonly encountered with low-level I/O:

```py
data = b'Hello World\r\n'
len(data)                         # 13
data[0:5]                         # b'Hello'
data.replace(b'Hello', b'Cruel')  # b'Cruel World\r\n'
data[0]   # 72 (ASCII code for 'H')
text = data.decode('utf-8') # bytes -> text
data = text.encode('utf-8') # text -> bytes
```

The 'utf-8' argument specifies a character encoding. Other common values include 'ascii' and 'latin1'.

### Raw strings

Raw strings are string literals with an uninterpreted backslash.

```py
>>> rs = r'c:\newdata\test' # Raw (uninterpreted backslash)
>>> rs
'c:\\newdata\\test'
```

The string is the literal text enclosed inside, exactly as typed. This is useful in situations where the backslash has special significance. Example: filename, regular expressions, etc.

### f-Strings (python3.6 or newer)

A string with formatted expression substitution.

```py
>>> name = 'IBM'
>>> shares = 100
>>> price = 91.1
>>> a = f'{name:>10s} {shares:10d} {price:10.2f}'
>>> a
'       IBM        100      91.10'
>>> b = f'Cost = ${shares*price:0.2f}'
>>> b
'Cost = $9110.00'
>>>
```

Formatting

```
d       Decimal integer
b       Binary integer
x       Hexadecimal integer
f       Float as [-]m.dddddd
e       Float as [-]m.dddddde+-xx
g       Float, but selective use of E notation s String
c       Character (from integer)
```

Common modifiers

```
:>10d   Integer right aligned in 10-character field
:<10d   Integer left aligned in 10-character field
:^10d   Integer centered in 10-character field
:0.2f   Float with 2 digit precision
```

## Some tips on lists

`l.sort()` sorts `l` in-place and doesn't return anything. `sorted(l)` keeps `l` unchanged and returns a new list.

Sort by descending order: `l.sort(reverse=True)`.

Lists have builtin method `l.count()`.

Caution: Be cautious whenever doing something like `l = [obj] * 5`. This means this list has 5 of the same object. Once `obj` is updated, all the elements in the list are updated.

`enumerate`: use `enumerate` more in loops, it gets the index and the item.

```py
names = ['Elwood', 'Jake', 'Curtis']
for i, name in enumerate(names):
    # Loops with i = 0, name = 'Elwood'
    # i = 1, name = 'Jake'
    # i = 2, name = 'Curtis'
```

`enumerate(sequence [, start = 0])`, `start` is optional. Example: `for lineno, line in enumerate(f, start=1):...`.

## Files: common idioms

Read an entire file all at once as a string.

```py
with open('foo.txt', 'rt') as file:
    data = file.read()
    # `data` is a string with all the text in `foo.txt`
```

Read a file line-by-line by iterating.

```py
with open(filename, 'rt') as file:
    for line in file:
        # Process the line
```

Write a file,

```py
with open('outfile', 'wt') as out:
    out.write('Hello World\n')
    ...
```

Redirect the print function,

```py
with open('outfile', 'wt') as out:
    print('Hello World', file=out)
    ...
```

## Catching and handling exceptions

Exceptions can be caught and handled.

To catch, use the `try - except` statement.

```py
for line in f:
    fields = line.split()
    try:
        shares = int(fields[1])
    except ValueError:
        print("Couldn't parse", line)
    ...
```

It is often difficult to know exactly what kinds of errors might occur in advance depending on the operation being performed. For better or for worse, exception handling often gets added after a program has unexpectedly crashed (i.e., “oh, we forgot to catch that error. We should handle that!”).

To raise an exception, use the `raise` statement.

```py
raise RuntimeError('What a kerfuffle')
```

This will cause the program to abort with an exception traceback. Unless caught by a try-except block.

## Data structures

Tuples look like read-only lists. However, tuples are most often used for a single item consisting of multiple parts. Lists are usually a collection of distinct items, usually all of the same type.

A dictionary is mapping of keys to values. It’s also sometimes called a hash table or associative array.

To delete a value in a dictionary, use the `del` statement. `del s['date']`.

Set default value if key doesn't exist.

```py
name = d.get(key, default)
```

Dict and set are both unordered.

## `collections`

There's [collections](https://docs.python.org/3/library/collections.html) which has

```
ChainMap, namedtuple, deque, Counter,
OrderedDict, defaultdict, UserDict, UserList, UserString
```

### `Counter`: Counting into buckets

```py
portfolio = [
    ('GOOG', 100, 490.1),
    ('IBM', 50, 91.1),
    ('CAT', 150, 83.44),
    ('IBM', 100, 45.23),
    ('GOOG', 75, 572.45),
    ('AA', 50, 23.15)
]

from collections import Counter
total_shares = Counter()
for name, shares, price in portfolio:
    total_shares[name] += shares

total_shares['IBM']     # 150
```

### One-many mapping: `defaultdict`

`defaultdict` ensures that every time you access a key you get a default value.

If we want to group the following list of tuples into key value pairs where key is `name` and value is a list of `(share, price)` tuples and default to an empty list, we do the following.

```py
portfolio = [
    ('GOOG', 100, 490.1),
    ('IBM', 50, 91.1),
    ('CAT', 150, 83.44),
    ('IBM', 100, 45.23),
    ('GOOG', 75, 572.45),
    ('AA', 50, 23.15)
]

from collections import defaultdict
holdings = defaultdict(list)
for name, shares, price in portfolio:
    holdings[name].append((shares, price))
holdings['IBM'] # [ (50, 91.1), (100, 45.23) ]
```

### `deque`: queue and stack

Problem: keep the last N elements.

```py
from collections import deque

history = deque(maxlen=N)
with open(filename) as f:
    for line in f:
        history.append(line)
        ...
```

## `zip` function

The `zip` function takes **multiple sequences** and makes an iterator that combines them.

```py
columns = ['name', 'shares', 'price']
values = ['GOOG', 100, 490.1 ]
pairs = zip(columns, values)
# ('name','GOOG'), ('shares',100), ('price',490.1)
```

To get the result you must iterate. You can use multiple variables to unpack the tuples as shown earlier.

```py
for column, value in pairs:
    ...
```

A common use of zip is to create key/value pairs for constructing dictionaries.

```py
d = dict(zip(columns, values))
```

## Object model

In Python, everything is an object. There's danger dealing with mutable objects.

**Assignment operations never make a copy of the value being assigned. All assignments are merely reference copies (or pointer copies if you prefer).**

Remember: **Variables are names, not memory locations**.

### Identity and References

Use the `is` operator to check if two values are exactly the same object.

```py
>>> a = [1,2,3]
>>> b = a
>>> a is b
True
```

**`is` compares the object identity (an integer). The identity can be obtained using `id()`.**

```py
>>> id(a)
3588944
>>> id(b)
3588944
```

### Shallow copies: beware!

Lists and dicts have methods for copying.

```py
>>> a = [2,3,[100,101],4]
>>> b = list(a) # Make a copy
>>> a is b
False
```

It’s a new list, but the list **items are shared**!!

```py
>>> a[2].append(102)
>>> b[2]
[100,101,102]
>>>
>>> a[2] is b[2]
True
```

<img src="{{ site.baseurl }}/images/misc/pyshallowcopy.png" alt="" align="middle"/>

### Deep copies

Sometimes you need to **make a copy of an object and all the objects contained within it**.

You can use the `copy` module for this: `copy.deepcopy()`

```py
>>> a = [2,3,[100,101],4]
>>> import copy
>>> b = copy.deepcopy(a)
>>> a[2].append(102)
>>> b[2]
[100,101]
>>> a[2] is b[2]
False
```

### Type checking

How to tell if an object is a specific type.

```py
if isinstance(a, list):
    print('a is a list')
```

Checking for one of many possible types.

```py
if isinstance(a, (list,tuple)):
    print('a is a list or tuple')
```

*Caution: Don’t go overboard with type checking. It can lead to excessive code complexity. Usually you’d only do it if doing so would prevent common mistakes made by others using your code.*

### Everything is an object

Numbers, strings, lists, functions, exceptions, classes, instances, etc. are all objects.

It means that all objects that can be named can be passed around as data, placed in containers, etc., without any restrictions.

There are no special kinds of objects. Sometimes it is said that all objects are “first-class”.

```py
>>> import math
>>> items = [abs, math, ValueError ]
>>> items
[<built-in function abs>,
  <module 'math' (builtin)>,
  <type 'exceptions.ValueError'>]
>>> items[0](-45)
45
>>> items[1].sqrt(2)
1.4142135623730951
>>> try:
        x = int('not a number')
    except items[2]:
        print('Failed!')
Failed!
```

## Program organization

### Function best practices

Define functions at the top of a script, put all of the code related to a single task all in one function.

Functions need to be modular and predictable.

Write docstring, describe what the function does in one sentence, and add information per argument.

Add optional **type hints** to function definitions.

```py
def read_prices(filename: str) -> dict:
    '''
    Read prices from a CSV file of name,price data
    '''
    prices = {}
    with open(filename) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            prices[row[0]] = float(row[1])
    return prices
```

These type hints do nothing operationally. Yet IDEs can use them to give hints.

When calling a function with optional arguments, use keywords instead of just True or False.

```py
parse_data(data, False, True) # ????? NO

parse_data(data, ignore_errors=True)
parse_data(data, debug=True)
parse_data(data, debug=True, ignore_errors=True)
```

Always give short but meaningful names.

Regarding variable scope, **remember: All assignments in functions are local.**

If you must update a global variable, use the `global` keyword.

```py
name = 'Dave'

def spam():
    global name
    name = 'Guido' # Changes the global name above
```

**Actually, avoid global variable entirely if you can. If you want to modify a state outside of a function, use a class instead.**

Arguments passed into functions are references. **If a mutable object get passed in and you use its method to modify it, it will be modified in-place**.

But keep in mind, reassignment to an old variable name inside a function only modifies the name in the local scope, it doesn't change the original object:

```py
def foo(items):
    items.append(42)    # Modifies the input object in-place because .append()

a = [1, 2, 3]
foo(a)
print(a)                # [1, 2, 3, 42]

# VS
def bar(items):
    items = [4,5,6]    # Changes local `items` variable to point to a different object

b = [1, 2, 3]
bar(b)
print(b)                # [1, 2, 3]
```

### Error handling

Python doesn't check data type or values, if there's any error, it will appear at run time as an exception.

```py
# raise
def authenticate(name):
    if name not in authorized:
        raise RuntimeError(f'{name} not authorized')

# try - except
try:
    authenticate(username)
except RuntimeError as e: # error msg f'{name} not authorized' is in `e`
    print(e) # HANDLE THE EXCEPTION HERE!!!
    ...
statements        # Resumes execution here after handling exception
statements        # And continues here
...
```

Some Python builtin exceptions for indicating what is wrong:

```py
ArithmeticError
AssertionError
EnvironmentError
EOFError
ImportError
IndexError
KeyboardInterrupt
KeyError
MemoryError
NameError
ReferenceError
RuntimeError
SyntaxError
SystemError
TypeError
ValueError
...
```

Handling multiple errors in different ways:

```py
try:
  ...
except LookupError as e:
  ...
except RuntimeError as e:
  ...
except IOError as e:
  ...
except KeyboardInterrupt as e:
  ...
```

Grouping them and handle the same way:

```py
try:
  ...
except (IOError,LookupError,RuntimeError) as e:
  ...
```

To catch all errors:

```py
try:
    ...
except Exception:       # DANGER. See below
    print('An error occurred')
```

**This is a bad idea because you’ll have no idea why it failed.**

Recommended approach: `as e`

```py
try:
    go_do_something()
except SomeSpecificException as e:
    print('Computer says no. Reason :', e)
    raise
```

Using `raise` allows you to take action (e.g. logging) and pass the exception to the caller.

The finally clause:

```py
lock = Lock()
...
lock.acquire()

try:
    ...
finally:
    lock.release() # this will ALWAYS be executed. With and w/o exception.
```

Finally is commonly used to safely manage resources (especially locks, files, etc.).

However, the best practice is to use `with` and avoid this approach.

```py
lock = Lock()
with lock:
    # lock acquired
    ...
# lock released
```

### Modules

Any `.py` file is a module. The `import` statement loads and executes a module.

A module is a collection of named values and is sometimes said to be a **namespace**. The names are all of the **global variables** and **functions** defined in the source file. After importing, the module name is used as a prefix. Hence the namespace.

```py
import foo

a = foo.grok(2)
b = foo.spam('Hello')
...
```

**The module name is directly tied to the file name (foo -> foo.py).**

Modules are isolated. `foo.x` and `bar.x` are different:

```py
# foo.py
x = 42
# bar.py
x = 37
```

Global variables are always bound to the enclosing module (same file). Each source file is its own little universe.

#### Module execution

When a module is imported, **all of the statements in the module execute one after another until the end of the file is reached**. The contents of the module namespace are all of the global names that are still defined at the end of the execution process. **If there are scripting statements that carry out tasks in the global scope (printing, creating files, etc.) you will see them run on import.**

#### Module Loading

Each module loads and executes **only once**. Note: **Repeated imports just return a reference to the previously loaded module**.

`sys.modules` is a `dict` of all loaded modules.

```py
>>> import sys
>>> sys.modules.keys()
['copy_reg', '__main__', 'site', '__builtin__', 'encodings', 'encodings.encodings', 'posixpath', ...]
```

**Caution**: A common confusion arises if you repeat an import statement after changing the source code for a module. **Because of the module cache sys.modules, repeated imports always return the previously loaded module – *even if a change was made*! The safest way to load modified code into Python is to quit and restart the interpreter/kernel!!**

#### Locating modules

Python consults a path list (sys.path) when looking for modules. The current working directory is usually first.

```py
>>> import sys
>>> sys.path
[
  '',
  '/usr/local/lib/python36/python36.zip',
  '/usr/local/lib/python36',
  ...
]
```

`sys.path` contains the **search paths**. You can manually adjust if you need to:

```py
import sys
sys.path.append('/project/foo/pyfiles')
```

Paths can also be added via environment variables.

```py
% env PYTHONPATH=/project/foo/pyfiles python3
Python 3.6.0 (default, Feb 3 2017, 05:53:21)
[GCC 4.2.1 Compatible Apple LLVM 8.0.0 (clang-800.0.38)]
>>> import sys
>>> sys.path
['','/project/foo/pyfiles', ...]
```

As a general rule, it should not be necessary to manually adjust the module search path. However, it sometimes arises if you’re trying to import Python code that’s in an unusual location or not readily accessible from the current working directory.

#### Main module

In many programming languages, there is a concept of a `main` function or method. This is the first function that executes when an application is launched.

Python has no main function or method. Instead, there is a **main module**. **The main module is the source file that runs first**. Whatever file you give to the interpreter at startup becomes main. It doesn’t matter the name.

Any Python file can either run as main or as a library import:

```py
bash % python3 prog.py # Running as main

import prog   # Running as library import
```

In both cases, `__name__` is the name of the module. However, it will only be set to `__main__` if running as main.

**Usually, you don’t want statements that are part of the main program to execute on a library import**. So, it’s common to have an `if`-check in code that might be used either way.

```py
# prog.py
...
if __name__ == '__main__':
    # Running as the main program ...
    statements
    ...
```

Here is common Python program template:

```py
# prog.py
# Import statements (libraries)
import modules

# Functions
def spam():
    ...

def blah():
    ...

# Main function
def main():
    ...

if __name__ == '__main__':
    main()
```

When used as a CLI tool, like `bash % python report.py portfolio.csv prices.csv`, the list of arguments is in `sys.argv`:

```py
# In the previous bash command
sys.argv # ['report.py, 'portfolio.csv', 'prices.csv']
```

`sys.stdout`, `sys.stderr` and `sys.stdin` are files that work the same way as normal files. By default, print is directed to `sys.stdout`. Input is read from `sys.stdin`. Tracebacks and errors are directed to `sys.stderr`.

`stdio` could be connected to terminals, files, pipes, etc.

```
bash % python3 prog.py > results.txt
# or
bash % cmd1 | python3 prog.py | cmd2
```

#### Environment variables

Environment variables are set in the shell.

```
bash % setenv NAME dave
bash % setenv RSH ssh
bash % python prog.py
```

`os.environ` is a dictionary that contains these values.

```py
import os

name = os.environ['NAME'] # 'dave'
```

#### Program Exit

Program exit is handled through exceptions. A non-zero exit code indicates an error.

```py
raise SystemExit
raise SystemExit(exitcode)
raise SystemExit('Informative message')
# Or
import sys
sys.exit(exitcode)
```

#### The `!#` line

On Unix, the `#!` shebang line can launch a script as Python. Add the following to the first line of your script file.

```
#!/usr/bin/env python3
```

It requires the executable permission.

```
bash % chmod +x prog.py
# Then you can execute
bash % prog.py
... output ...
```

*Note: The Python Launcher on Windows also looks for the #! line to indicate language version.*

### Design generic and flexible functions

Compare the following two versions of a function

```py
# VERSION 1
# Provide a filename
def read_data(filename):
    records = []
    with open(filename) as f:
        for line in f:
            ...
            records.append(r)
    return records

d = read_data('file.csv')

# VERSION 2
# Provide lines
def read_data(lines):
    records = []
    for line in lines:
        ...
        records.append(r)
    return records

with open('file.csv') as f:
    d = read_data(f)
```

Version 2 is better because it's more generic, it takes in any iterable.

[Duck Typing](https://en.wikipedia.org/wiki/Duck_typing) is a computer programming concept to determine whether an object can be used for a particular purpose. It is an application of the duck test.

> If it looks like a duck, swims like a duck, and quacks like a duck, then it probably is a duck.

Code libraries are often better served by embracing flexibility. Don’t restrict your options. With great flexibility comes great power.

## Reference

<https://dabeaz-course.github.io/practical-python/Notes>
