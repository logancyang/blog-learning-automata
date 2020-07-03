---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Review] Part III: Generators and Advanced Topics"
comments: true
---

## Generators

### Iteration protocol

Many objects support iteration:

```py
a = 'hello'
for c in a: # Loop over characters in a
    ...

b = { 'name': 'Dave', 'password':'foo'}
for k in b: # Loop over keys in dictionary
    ...

c = [1,2,3,4]
for i in c: # Loop over items in a list/tuple
    ...

f = open('foo.txt')
for x in f: # Loop over lines in a file
    ...
```

What happens under the hood of a `for` loop?

```py
for x in obj:
    ...

# Is equivalent to
_iter = obj.__iter__()        # Get iterator object
while True:
    try:
        x = _iter.__next__()  # Get next item
    except StopIteration:     # No more items
        break
    # statements ...
```

All objects that support `for` loop implement this low level iteration protocol. This is a manual iteration through a list:

```py
>>> x = [1,2,3]
>>> it = x.__iter__()
>>> it
<listiterator object at 0x590b0>
>>> it.__next__()
1
>>> it.__next__()
2
>>> it.__next__()
3
>>> it.__next__()
Traceback (most recent call last):
File "<stdin>", line 1, in ? StopIteration
```

#### Support iteration in your custom object

Knowing about iteration is useful if you want to add it to your own objects. For example, making a custom container.

```py
class Portfolio:
    def __init__(self):
        self.holdings = []

    def __iter__(self):
        return self.holdings.__iter__()
    ...

port = Portfolio()
for s in port:
    ...
```

For container objects, supporting iteration, indexing, containment, and other kinds of operators is an important part of being Pythonic.

*Side note: `__contains__()` is a function for the `in` check, for example:*

```py
def __contains__(self, name):
        return any([s.name == name for s in self._holdings])
```

#### `next()` built-in function

The `next()` built-in function is a shortcut for calling the `__next__()` method of an iterator. Try using it on a file:

```py
>>> f = open('Data/portfolio.csv')
>>> f.__iter__()    # Note: This returns the file itself
<_io.TextIOWrapper name='Data/portfolio.csv' mode='r' encoding='UTF-8'>
>>> next(f)
'name,shares,price\n'
>>> next(f)
'"AA",100,32.20\n'
>>> next(f)
'"IBM",50,91.10\n'
```

### Customizing iteration

Now we look at how we can generalize iteration using a **generator** function.

Suppose you wanted to create your own custom iteration pattern.

For example, a countdown.

```py
>>> for x in countdown(10):
...   print(x, end=' ')
...
10 9 8 7 6 5 4 3 2 1
```

There is an easy way to do this.

#### Generator

A generator is a function that defines iteration.

```py
def countdown(n):
    while n > 0:
        yield n
        n -= 1

>>> for x in countdown(10):
...   print(x, end=' ')
...
10 9 8 7 6 5 4 3 2 1
```

**Definition: A generator is any function that uses the `yield` statement.**

**The behavior of generators is different than a normal function. Calling a generator function creates a generator object. It does not immediately execute the function.**

```py
def countdown(n):
    # Added a print statement
    print('Counting down from', n)
    while n > 0:
        yield n
        n -= 1

>>> x = countdown(10)
# There is NO PRINT STATEMENT
>>> x
# x is a generator object
<generator object at 0x58490>
```

The function only executes on `__next__()` call.

```py
>>> x = countdown(10)
>>> x
<generator object at 0x58490>
>>> x.__next__()
Counting down from 10
10
>>>
```

**`yield` produces a value, but *suspends the function execution*. The function resumes on next call to `__next__()`.**

```py
>>> x.__next__()
9
>>> x.__next__()
8
```

When the generator finally returns, the iteration raises an error.

```py
>>> x.__next__()
1
>>> x.__next__()
Traceback (most recent call last):
File "<stdin>", line 1, in ? StopIteration
```

This means **a generator function implements the same low-level protocol that the for statements uses on lists, tuples, dicts, files, etc.**

#### Generator example: find matching substring from lines in file

```py
>>> def filematch(filename, substr):
        with open(filename, 'r') as f:
            for line in f:
                if substr in line:
                    yield line

>>> for line in open('Data/portfolio.csv'):
        print(line, end='')

name,shares,price
"AA",100,32.20
"IBM",50,91.10
"CAT",150,83.44
"MSFT",200,51.23
"GE",95,40.37
"MSFT",50,65.10
"IBM",100,70.44
>>> for line in filematch('Data/portfolio.csv', 'IBM'):
        print(line, end='')

"IBM",50,91.10
"IBM",100,70.44
>>>
```

#### Generator example: monitoring a streaming data source

Suppose there is a running program that keeps writing to `Data/stocklog.csv` in realtime. We use the code below to monitor the stream.

```py
# follow.py
import os
import time

f = open('Data/stocklog.csv')
f.seek(0, os.SEEK_END)   # Move file pointer 0 bytes from end of file

while True:
    line = f.readline()
    if line == '':
        time.sleep(0.1)   # Sleep briefly and retry
        continue
    fields = line.split(',')
    name = fields[0].strip('"')
    price = float(fields[1])
    change = float(fields[4])
    if change < 0:
        print(f'{name:>10s} {price:>10.2f} {change:>10.2f}')
```

This `while True` loop along with some `if` checks and small `sleep` time keeps checking the end of the file. `readline()` will either return new data or an empty string, so if we get empty string we continue to the next retry.

It is just like the Unix `tail -f` command that is used to watch a log file.

### Producers, consumers and pipelines

Generators are a useful tool for setting various kinds of producer/consumer problems and dataflow pipelines.

#### Producer-Consumer Problems

```py
# Producer
def follow(f):
    ...
    while True:
        ...
        yield line        # Produces value in `line` below
        ...

# Consumer
for line in follow(f):    # Consumes vale from `yield` above
    ...
```

`yield` produces values that `for` consumes.

#### Generator Pipelines

You can use this aspect of generators to set up processing pipelines (like Unix pipes).

*producer → processing → processing → consumer*

Processing pipes have an initial data producer, some set of intermediate processing stages and a final consumer.

- The producer is typically a generator. Although it could also be a list of some other sequence. `yield` feeds data into the pipeline.
- Intermediate processing stages simultaneously consume and produce items. They might modify the data stream. They can also filter (discarding items).
- Consumer is a for-loop. It gets items and does something with them.

```py
"""
producer → processing → processing → consumer
"""
def producer():
    ...
    yield item          # yields the item that is received by the `processing`
    ...

def processing(s):
    for item in s:      # Comes from the `producer`
        ...
        yield newitem   # yields a new item
        ...

def consumer(s):
    for item in s:      # Comes from the `processing`
        ...

"""
To actually use it and setup the pipeline
"""
a = producer()
b = processing(a)
c = consumer(b)
```

**You can create various generator functions and chain them together to perform processing involving data-flow pipelines. In addition, you can create functions that package a series of pipeline stages into a single function call**

#### Generator expressions

Generator expressions are like list comprehension, except that generator expressions use `()` instead of `[]`.

```py
>>> b = (2*x for x in a)
>>> b
<generator object at 0x58760>
>>> for i in b:
...   print(i, end=' ')
...
2 4 6 8
```

Differences with List Comprehensions.

- Does not construct a list.
- Only useful purpose is iteration.
- **Once consumed, can’t be reused.**

General syntax: `(<expression> for i in s if <conditional>)`.

It can also serve as a function argument: `sum(x*x for x in a)`.

It can be applied to any iterable.

```py
>>> a = [1,2,3,4]
>>> b = (x*x for x in a)
>>> c = (-x for x in b)
>>> for i in c:
...   print(i, end=' ')
...
-1 -4 -9 -16
```

**The main use of generator expressions is in code that performs some calculation on a sequence, but only uses the result once.** For example, strip all comments from a file.

```py
f = open('somefile.txt')
lines = (line for line in f if not line.startswith('#'))
for line in lines:
    ...
f.close()
```

**With generators, the code runs faster and uses little memory. It’s like *a filter applied to a stream*.**

#### Why Generators
- Many problems are much more clearly expressed in terms of iteration.
  - Looping over a collection of items and performing some kind of operation (searching, replacing, modifying, etc.).
  - Processing pipelines can be applied to a wide range of data processing problems.
- Better memory efficiency.
  - Only produce values when needed.
  - Contrast to constructing giant lists.
  - Can operate on streaming data
- Generators encourage code reuse
  - Separates the iteration from code that uses the iteration
  - You can build a toolbox of interesting iteration functions and mix-n-match.

#### `itertools` module

The `itertools` is a library module with various functions designed to help with iterators/generators.

```py
itertools.chain(s1,s2)
itertools.count(n)
itertools.cycle(s)
itertools.dropwhile(predicate, s)
itertools.groupby(s)
itertools.ifilter(predicate, s)
itertools.imap(function, s1, ... sN)
itertools.repeat(s, n)
itertools.tee(s, ncopies)
itertools.izip(s1, ... , sN)
```

All functions process data iteratively. They implement various kinds of iteration patterns.

## Advanced topics



## Reference

<https://dabeaz-course.github.io/practical-python/Notes>
