---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "Python Essentials: Mental Models of Expert Level Features"
comments: true
---

## Metaclass: problem

Suppose there is an infra team that writes library code for other developers to use, and there is user code for business logics.

Here we have `library.py` from the infra team, `user.py` from the user facing team:

```py
# library.py
class Base:
    def foo(self):
        return 'foo'

# user.py
from library import Base

class Derived(Base):
    def bar(self):
        return self.foo
```

Suppose we are on the user facing team and we can't change the library code. What if the `foo` method gets removed? We don't want our `bar` method to break at runtime, we want to catch this before runtime production environment. How to do this easily?

We could add a line above the `Derived` class,

```py
assert hasattr(Base, "foo"), "You broke it!"
```

This enforces a constraint on the Base class.

Now let's consider the scenario below where we are in the infra team and are responsible for the library code. We need to make sure the business logic team implement the `bar` method.

```py
# library.py
class Base:
    def foo(self):
        return self.bar()


# user.py
from library import Base

class Derived(Base):
    def bar(self):
        return 'bar'
```

We can't do the same thing as before `assert hasattr(Derived, "bar"), "..."`. Try-except also doesn't work because it only catches the error at runtime, not before.

Python is a "protocol-oriented" language. In C++ and Java, a class definition is not executable code. But in Python it is.

Python has this `__build_class__()` method that lets you check things at the time when the class is being built.

```py
old_bc = __build_class__
def my_bc(func, name, base=None, **kw):
    if base is Base:
        print("Check if bar method defined")
    if base is not None:
        return old_bc(func, name, base, **kw)
    return old_bc(func, name, **kw)

import builtins
builtins.__build_class__ = my_bc

"""
Running the code above:

    Check if bar method defined
"""
```

This isn't typically what we do for the purpose of checking whether the library code is going to break when used in user code. This is just to show that in Python we can hook into the processes of building classes, defining functions, importing modules, and do what we want.

This pattern exists but we don't use it for this purpose, people use 2 fundamental features of Python to solve this problem of enforcing constraints.

The first one is Metaclass.

## Metaclass: solution

A metaclass is just a class derived from `type` class that allows you to intercept derived types.

```py
class BaseMeta(type):
    def __new__(cls, name, bases, body):
        print('BaseMeta.__new__', cls, name, bases, body)
        # To prevent user class without `bar` method
        if not 'bar' in body:
            raise TypeError("Bad user class")
        return super().__new__(cls, name, bases, body)

class Base(metaclass=BaseMeta):
    def foo(self):
        return self.bar()
```

**Metaclass is way to enforce constraints on the derived classes, e.g. user code, in the base classes, i.e. library code.**

*Checkout `collections.abc`'s ([doc](https://docs.python.org/3/library/collections.abc.html)) `abc` metaclass which has decorators such as `@abstractmethod` so we don't have to write metaclass ourselves!*

## Decorators

*Again, Python is a "live" language as in that function definitions, class definitions are executable code that gets executed line by line at runtime.*

The function decorator is a very important pattern in Python to simplify user code. It makes quick function wrappers. Say we want to time a function, we can do:

```py
from time import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print("elapsed time:", after - before)
        return rv
    return wrapper_func

@timeit
def add(x, y):
    return x + y

"""
Decorator is merely a syntax, it is equivalent to
"""
add = timeit(add)
```
### functools.wraps

`functools.wraps` is simply a way to preserve the name and docstring of the the original function that is being wrapped. Without it, the wrapped function would lose its original metadata.

```py
from functools import wraps

def logged(func):
    @wraps(func)
    def with_logging(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return with_logging

@logged
def f(x):
   """does some math"""
   return x + x * x

print(f.__name__)  # prints 'f'
print(f.__doc__)   # prints 'does some math'
```

Note that not all decorators wrap functions, some of them just "register" the name of the input function, for example. In that case we don't need to use `functools.wraps`. **But usually we should use it for decorators**.

### Decorator with parameters: the `partial` trick

One way of thinking about decorators with parameters is

```py
@decorator
def foo(*args, **kwargs):
    pass
```

translates to

```py
foo = decorator(foo)
```

So if the decorator had parameters,

```py
@decorator_with_params(param)
def foo(*args, **kwargs):
    pass
```

translates to

```py
foo = decorator_with_params(param)(foo)
```

To make this work, we need to write the decorator function as follows,

```py
from functools import partial, wraps

def _outer(func, params):
    # magic sauce to lift the name and doc of the function
    @wraps(func)
    def inner(*args, **kwargs):
        #do stuff here, for eg.
        print ("decorator params: %s" % str(params))
        return func(*args, **kwargs)
    return inner

# THIS IS THE KEY TRICK TO TAKE PARAMS INTO DECORATORS
real_decorator = partial(_outer, params=param)

@real_decorator
def bar(*args, **kwargs):
    pass
```

*The line `real_decorator = partial(_outer, params=param)` is the key trick to take parameters into a decorator function!*

## Generators: two mental models

There are two important mental models for generators.

### Laziness

A generator avoids eager execution and gives you the item you want one by one. This lets you avoid waiting for the entire loop to finish and storing all the items in it.

```py
for i in range(10):
    sleep(0.5)
    yield i
```

### Sequencing

A generator **can enforce sequences in execution**. It enables interleaving of **coroutines**: some library code runs, then some user code, then some library code...

Say we have an API that needs run some methods **in order**.

```py
"""
In this scenario, there is no enforcement of order
"""
class Api:
    def run_this_first(self):
        first()
    def run_this_second(self):
        second()
    def run_this_last(self):
        last()

"""
In this scenario, last() will never be run before second(), by
using the generator.
"""
def api():
    first()
    yield
    second()
    yield
    last()
```

## Context managers

Simply put, a Context Manager makes sure a pair of commands: an initial action and a final action are always executed.

Here's what a context manager is under the hood.

```py
with ctx() as x:
    pass

# <=>

x = ctx().__enter__()
try:
    pass
finally:
    x.__exit__()
```

What do we need this? For example, when we do something in a database and would like to drop table when we are done:

```py
from sqlite3 import connect

with connect('test.db') as conn:
    cur = conn.cursor()
    cur.execute('create table points(x int, y int)')
    cur.execute('insert into points (x, y) values(1, 1)')
    cur.execute('insert into points (x, y) values(1, 2)')
    cur.execute('insert into points (x, y) values(2, 1)')
    for row in cur.execute('select x, y from points'):
        print(row)
    cur.execute('drop table points')
```

We can write our own context manager to achieve this by:

```py
class temptable:
    def __init__(self, cur):
        self.cur = cur
    def __enter__(self):
        print('__enter__')
        self.cur.execute('create table points(x int, y int)')
    def __exit__(self):
        print('__exit__')
        self.cur.execute('drop table points')

with connect('test.db') as conn:
    cur = conn.cursor()
    with temptable(cur):
        cur.execute('insert into points (x, y) values(1, 1)')
        cur.execute('insert into points (x, y) values(1, 2)')
        cur.execute('insert into points (x, y) values(2, 1)')
        for row in cur.execute('select x, y from points'):
            print(row)
```

This is a bit better, we can enforce running `__enter__` before `__exit__` by using context manager, i.e. in this case we enforce `create table` before `drop table`.

Now we notice we have **sequencing** in execution because one must run before another. This reminds us about **generators**.

We can refactor,

```py
# THIS IS A GENERATOR WITH INPUT `cur`
def temptable(cur):
    cur.execute('create table points(x int, y int)')
    yield
    cur.execute('drop table points')

# THIS IS A CALLABLE CLASS WITH A GENERATOR INPUT
class Contextmanager:
    def __init__(self, gen):
        self.gen = gen

    def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __enter__(self):
        print('__enter__')
        self.gen_instance = self.gen(*self.args, **self.kwargs)
        next(self.gen_instance)

    def __exit__(self, *args):
        print('__exit__')
        next(self.gen_instance, None)

temptable = Contextmanager(temptable)

with connect('test.db') as conn:
    cur = conn.cursor()
    with temptable(cur):
        cur.execute('insert into points (x, y) values(1, 1)')
        cur.execute('insert into points (x, y) values(1, 2)')
        cur.execute('insert into points (x, y) values(2, 1)')
        for row in cur.execute('select x, y from points'):
            print(row)
```

The line `temptable = Contextmanager(temptable)` reminds us of **decorators**!

Fortunately, we don't need to write `Contextmanager` class ourselves, there is this library called `contextlib` which already provides it.

```py
from sqlite3 import connect
from contextlib import contextmanager


# THIS IS A GENERATOR WITH INPUT `cur`
# @contextmanager TURNS A GENERATOR INTO A CONTEXT MANAGER!
@contextmanager
def temptable(cur):
    cur.execute('create table points(x int, y int)')
    try:
        yield
    finally:
        cur.execute('drop table points')


with connect('test.db') as conn:
    cur = conn.cursor()
    with temptable(cur):
        cur.execute('insert into points (x, y) values(1, 1)')
        cur.execute('insert into points (x, y) values(1, 2)')
        cur.execute('insert into points (x, y) values(2, 1)')
        for row in cur.execute('select x, y from points'):
            print(row)
```

`@contextmanager` turns a generator into a context manager! This is a really useful pattern for creating a context manager for a pair of commands.

This last example combined the 3 core features of Python together:
- **decorator**: syntactic sugar for function wrapping
- **generator**: avoid eagerness, save resources, force sequencing
- **context manager**: force paired commands

## Summary

Python is a language oriented around *protocols*. There are ways to implement these protocols on any objects using "dunder" methods. If you forget how to use these methods, just google "Python data model".

## Reference

- [James Powell: So you want to be a Python expert? PyData Seattle 2017](https://www.youtube.com/watch?v=cKPlPJyQrt4) (~2hr)
- `functools.wraps`: <https://stackoverflow.com/questions/308999/what-does-functools-wraps-do>
- The `partial` trick for decorators with parameters: <https://stackoverflow.com/a/25827070/2280673>
