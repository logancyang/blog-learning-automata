---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "Python Essentials: Reviewing The Basics"
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



