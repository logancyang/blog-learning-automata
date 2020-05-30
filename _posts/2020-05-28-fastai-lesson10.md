---
toc: true
layout: post
description: fast.ai note series
categories: [note, fastai]
title: "FastAI Lesson 10: Looking Inside the Model"
comments: true
---

## Jeremy's starting comments: take it easy and make progress slowly

The amount of material in part II is meant to keep the student busy until the next version of the course. Don't expect to understand every thing in one go, digest them bit by bit.

It also covers the software engineering side. Jeremy's opinion is that **data scientists need to be good software engineers** as well.

<img src="{{ site.baseurl }}/images/fastai/from_foundations.png" alt="from_foundations" align="middle"/>

We will stick to using nothing but the foundation tools in the picture above to recreate the `fastai` library.

Next week we will develop a new library called `fastai.audio`! *(Exactly what I interested in: DL in audio and library development!)*

<img src="{{ site.baseurl }}/images/fastai/fastai_audio.png" alt="fastai_audio" align="middle"/>

Then we will get into seq2seq models, transformer, and more advanced vision models that requires setting up a DL box and doing experiments. `fastec2` library is useful for running experiments in AWS.

<img src="{{ site.baseurl }}/images/fastai/seq2seq.png" alt="seq2seq" align="middle"/>

<img src="{{ site.baseurl }}/images/fastai/adv_vision.png" alt="adv_vision" align="middle"/>

At last we will dive into Swift for DL.

## Revisiting Callbacks

Notebook: `05a_foundations`

### What is a callback

Callbacks are functions that get triggered at certain events. We pass the callback function object itself to a method.

### How to create a callback

```py
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        res += i*i
        sleep(1)
        if cb: cb(i)
    return res

def show_progress(epoch):
    print(f"Awesome! We've finished epoch {epoch}!")


slow_calculation(show_progress)
"""
Awesome! We've finished epoch 0!
Awesome! We've finished epoch 1!
Awesome! We've finished epoch 2!
Awesome! We've finished epoch 3!
Awesome! We've finished epoch 4!
"""
```

This callback `show_progress(epoch)` is just a function that's passed into the target function as an object. The target function has an expectation how to call it, i.e. passing in the # epoch in this case.

Since we are using it once, a better way to do this is to use the lambda function (similar to arrow functions in JavaScript ES6).

```py
slow_calculation(lambda o: print(f"Awesome! We've finished epoch {o}!"))
```

### Callback with more than one argument

```py
def make_show_progress(exclamation):
    # Leading "_" is generally understood to be "private"
    # `exclamation` is a context variable for _inner(epoch)
    # this is called closure
    def _inner(epoch):
        print(f"{exclamation}! We've finished epoch {epoch}!")
    return _inner

slow_calculation(make_show_progress("Nice!"))
"""
Nice!! We've finished epoch 0!
Nice!! We've finished epoch 1!
Nice!! We've finished epoch 2!
Nice!! We've finished epoch 3!
Nice!! We've finished epoch 4!
"""
```

`exclamation` is a context variable outside `_inner(epoch)`. This is called **closure**. This concept is prevalent in JS.


```py
f2 = make_show_progress("Terrific")

slow_calculation(f2)
"""
Terrific! We've finished epoch 0!
Terrific! We've finished epoch 1!
Terrific! We've finished epoch 2!
Terrific! We've finished epoch 3!
Terrific! We've finished epoch 4!
"""
```

### `partial` function

In Python, with `from functools import partial` we can make a new function that is the old function with predefined argument(s).

```py
from functools import partial

slow_calculation(partial(show_progress, "OK I guess"))
"""
OK I guess! We've finished epoch 0!
OK I guess! We've finished epoch 1!
OK I guess! We've finished epoch 2!
OK I guess! We've finished epoch 3!
OK I guess! We've finished epoch 4!
"""
```

`partial(func, arg, arg, ...)` takes positional arguments and knows how to set them in order.

### Callbacks as callable classes

Wherever we can use a closure to store a context, we can also use a class.

```py
class ProgressShowingCallback():
    def __init__(self, exclamation="Awesome"):
        self.exclamation = exclamation

    def __call__(self, epoch):
        """This is the part that makes the class callable as a function!"""
        print(f"{self.exclamation}! We've finished epoch {epoch}!")

cb = ProgressShowingCallback("Just super")
slow_calculation(cb)
"""
Just super! We've finished epoch 0!
Just super! We've finished epoch 1!
Just super! We've finished epoch 2!
Just super! We've finished epoch 3!
Just super! We've finished epoch 4!
"""
```

**In Python, `obj.__call__()` makes the `obj` callable as a function when used like `obj()`!**

### Python `*args` and `**kwargs`

A Python function puts the positional arguments into a tuple `args`, and the keyword arguments into a dictionary `kwargs`.

```py
def f(*args, **kwargs):
    print(f"args: {args}; kwargs: {kwargs}")

f(3, 'a', thing1="hello")
"""
args: (3, 'a'); kwargs: {'thing1': 'hello'}
"""
```

There are some downsides to using `args` and `kwargs`, e.g. when you check the signature of a function and you only see this and don't know what exactly is passed in. For example, if there's a typo in a parameter name, it's hard to track down.

Sometimes we do want to use them. For example, here the callback `cb` has two methods, one takes 1 argument and the other takes 2.

```py
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        if cb: cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb: cb.after_calc(i, val=res)
    return res


class PrintStepCallback():
    def __init__(self): pass

    def before_calc(self, *args, **kwargs):
        """In this case we don't care about what's passed in"""
        print(f"About to start")

    def after_calc (self, *args, **kwargs):
        print(f"Done step")
```

In this case we don't care about what's passed into the methods, `args` and `kwargs` are passed and not used.

If we remove them there will be an error when calling the methods with any arguments. With them, we can pass in whatever arguments at calling.

To make the methods do something with the input,

```py
class PrintStatusCallback():
    def __init__(self): pass
    def before_calc(self, epoch, **kwargs): print(f"About to start: {epoch}")
    def after_calc (self, epoch, val, **kwargs): print(f"After {epoch}: {val}")

slow_calculation(PrintStatusCallback())
"""
About to start: 0
After 0: 0
About to start: 1
After 1: 1
About to start: 2
After 2: 5
About to start: 3
After 3: 14
About to start: 4
After 4: 30
"""
```

Here we put `**kwargs` in case we want to add something in the future and make sure it doesn't break. If we pass in any unexpected positional arguments it *should* break.

### Callbacks: modifying behavior

#### Early stopping

We can modify the target function with the callback. Here's an example of early stopping using a callback.

```py
def slow_calculation(cb=None):
    res = 0
    for i in range(5):
        # `hasattr` avoids breaking if cb doesn't have the method
        if cb and hasattr(cb,'before_calc'): cb.before_calc(i)
        res += i*i
        sleep(1)
        if cb and hasattr(cb,'after_calc'):
            if cb.after_calc(i, res):
                print("stopping early")
                break
    return res

class PrintAfterCallback():
    def after_calc(self, epoch, val):
        print(f"After {epoch}: {val}")
        if val>10: return True

slow_calculation(PrintAfterCallback())
"""
After 0: 0
After 1: 1
After 2: 5
After 3: 14
stopping early
"""
```

#### Modifying the state

We can also directly modify the state of the object with the callback by passing the object into the callback.

```py
class SlowCalculator():
    def __init__(self, cb=None):
        self.cb, self.res = cb, 0

    def callback(self, cb_name, *args):
        if not self.cb:
            return
        cb = getattr(self.cb,cb_name, None)
        if cb:
            return cb(self, *args)

    def calc(self):
        for i in range(5):
            # We can use `__call__()` instead of `callback()` above,
            # then here becomes `self('before_calc', i)`
            self.callback('before_calc', i)
            self.res += i*i
            sleep(1)
            if self.callback('after_calc', i):
                print("stopping early")
                break


class ModifyingCallback():
    def after_calc(self, calc, epoch):
        print(f"After {epoch}: {calc.res}")
        if calc.res>10:
            return True
        # HERE WE MODIFIES `calc` object that is passed in!
        if calc.res<3:
            calc.res = calc.res*2

# Init the instance with the modifying callback
calculator = SlowCalculator(ModifyingCallback())
calculator.calc()
"""
After 0: 0
After 1: 1
After 2: 6
After 3: 15
stopping early
"""
calculator.res
"""
15
"""
```

## Revisiting Python Dunder Methods

The Python doc for its [data model](https://docs.python.org/3/reference/datamodel.html#object.__init__) has all the info about the special dunder methods `__xxx__()`.

A toy example,

```py
class SloppyAdder():
    def __init__(self,o): self.o=o
    def __add__(self,b): return SloppyAdder(self.o + b.o + 0.01)
    def __repr__(self): return str(self.o)

a = SloppyAdder(1)
b = SloppyAdder(2)
# `+` is overridden by __add__
a+b
"""
3.01
"""
```

Some examples:

- `__getitem__`
- `__getattr__`
- `__setattr__`
- `__del__`
- `__init__`
- `__new__`
- `__enter__`
- `__exit__`
- `__len__`
- `__repr__`
- `__str__`

## Fundamental ability of an engineer: browsing source code

Must know and practice how to do all these in [vscode](https://code.visualstudio.com/docs/editor/editingevolved),

- Jump to tag/symbol
- Jump to current tag
- Jump to library tags
- Go back
- Search
- Outlining / folding

Jeremy uses Vim because it's good for developing on remote machines. Nowadays vscode can use the ssh extension.

## Variance, covariance, and correlation

```py
"""
VARIANCE
"""
t = torch.tensor([1.,2.,4.,18])
m = t.mean()
(t-m).pow(2).mean()

"""
STANDARD DEVIATION
"""
(t-m).pow(2).mean().sqrt()

"""
MEAN ABSOLUTE DEVIATION
"""
(t-m).abs().mean()
```

Note that **Mean Absolute Deviation should be used more because it's more robust than the standard deviation for outliers**.

Notice that

```py
(t-m).pow(2).mean() == (t*t).mean() - (m*m)
```

This is equivalent to,

$$\operatorname{var}[X] = \operatorname{E}\left[X^2 \right] - \operatorname{E}[X]^2$$

**When we calculate the variance in code, we should use `(t*t).mean() - (m*m)` instead of the definition form because it's more efficient (doesn't require multiple passes).**

Similarly, we can calculate the covariance of two variables `t` and `v` by

```py
cov = (t*v).mean() - t.mean()*v.mean()
```

because,

$$\operatorname{cov}(X,Y) = \operatorname{E}{\big[(X - \operatorname{E}[X])(Y - \operatorname{E}[Y])\big]}$$
$$ = \operatorname{E}\left[X Y\right] - \operatorname{E}\left[X\right] \operatorname{E}\left[Y\right]$$

**Variance and covariance are the same thing, because variance is just the covariance of X with itself.**

Next we have correlation, or Pearson correlation coefficient,

$$\rho_{X,Y}= \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y}$$

In code,

```py
corr = cov / (t.std() * v.std())
```

**The correlation is just a scaled version of the covariance.**

**Remember: from now on, always write code for a math equation, not (just) the LaTeX!**






































