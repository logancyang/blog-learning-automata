---
toc: true
layout: post
description: Stepping up Python skills
categories: [note, python]
title: "[Python Review] Part II: Classes and Objects"
comments: true
---

## Class and object basics

### Overriding

Sometimes a class extends an existing method, but it wants to use the original implementation inside the redefinition. For this, use super() to call the old one:

```py
class Stock:
    ...
    def cost(self):
        return self.shares * self.price
    ...

class MyStock(Stock):
    def cost(self):
        # Check the call to `super`
        actual_cost = super().cost()
        return 1.25 * actual_cost
```

If `__init__` is redefined, it is essential to initialize the parent.

```py
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

class MyStock(Stock):
    def __init__(self, name, shares, price, factor):
        # Check the call to `super` and `__init__`
        super().__init__(name, shares, price)
        self.factor = factor

    def cost(self):
        return self.factor * super().cost()
```

### Organizing inheritance

Inheritance is sometimes used to organize related objects. Think of a logical hierarchy or taxonomy. However, **a more common (and practical) usage is related to making reusable or extensible code**. For example, **a framework might define a base class and instruct you to customize it**.

```py
class CustomHandler(TCPHandler):
    def handle_request(self):
        ...
        # Custom processing
```

The base class contains some general purpose code. Your class inherits and customized specific parts.

### “is a” relationship

Inheritance establishes a type relationship.

```py
class Shape:
    ...

class Circle(Shape):
    ...

>>> c = Circle(4.0)
>>> isinstance(c, Shape)
True
```

**Important: Ideally, any code that worked with instances of the parent class will also work with instances of the child class.**

### `object` base class

If a class has no parent, you sometimes see `object` used as the base. `object` is the parent of all objects in Python.

### Multiple Inheritance

You can inherit from multiple classes by specifying them in the definition of the class.

```py
class Mother:
    ...

class Father:
    ...

class Child(Mother, Father):
    ...
```

The class Child inherits features from both parents. There are some rather tricky details. **Don’t do it unless you know what you are doing**.

### Special methods

There are dozens of `__xxx__` methods in Python.

`__str__()` is used to create a nice printable output.

`__repr__()` is used to create a more detailed representation for programmers.

*Note: The convention for __repr__() is to return a string that, when fed to eval(), will recreate the underlying object. If this is not possible, some kind of easily readable representation is used instead.*

#### Special dunder methods for math

```py
a + b       a.__add__(b)
a - b       a.__sub__(b)
a * b       a.__mul__(b)
a / b       a.__truediv__(b)
a // b      a.__floordiv__(b)
a % b       a.__mod__(b)
a << b      a.__lshift__(b)
a >> b      a.__rshift__(b)
a & b       a.__and__(b)
a | b       a.__or__(b)
a ^ b       a.__xor__(b)
a ** b      a.__pow__(b)
-a          a.__neg__()
~a          a.__invert__()
abs(a)      a.__abs__()
```

#### Special dunder methods for item access

```py
len(x)      x.__len__()
x[a]        x.__getitem__(a)
x[a] = v    x.__setitem__(a,v)
del x[a]    x.__delitem__(a)

# Implement a sequence
class Sequence:
    def __len__(self):
        ...
    def __getitem__(self,a):
        ...
    def __setitem__(self,a,v):
        ...
    def __delitem__(self,a):
        ...
```

### Bound method

A method that has not yet been invoked by the function call operator `()` is known as a **bound method**. It operates on the instance where it originated.

```py
>>> s = Stock('GOOG', 100, 490.10)
>>> s
<Stock object at 0x590d0>
>>> c = s.cost
>>> c
<bound method Stock.cost of <Stock object at 0x590d0>>
>>> c()
49010.0
```

Bound methods are often a source of careless non-obvious errors: you simply forgot to add `()`.

### Attribute access

There is an alternative way to access, manipulate and manage attributes.

```py
getattr(obj, 'name')          # Same as obj.name
setattr(obj, 'name', value)   # Same as obj.name = value
delattr(obj, 'name')          # Same as del obj.name
hasattr(obj, 'name')          # Tests if attribute exists
```

Example:

```py
if hasattr(obj, 'x'):
    x = getattr(obj, 'x'):
else:
    x = None
```

Note: `getattr()` also has a useful default value `*arg`.

```py
x = getattr(obj, 'x', None)
```

### Defining new exceptions

User defined exceptions are defined by classes. **Exceptions always inherit from `Exception`. Usually they are empty classes. Use `pass` for the body.**

```py
class NetworkError(Exception):
    pass
```

You can make your own hierarchy of exceptions:

```py
class AuthenticationError(NetworkError):
     pass

class ProtocolError(NetworkError):
    pass
```

## Inner workings of Python objects

Programmers coming from other programming languages often find Python’s notion of classes lacking in features. *For example, there is no notion of access-control (e.g., private, protected), the whole `self` argument feels weird, and frankly, working with objects sometimes feel like a “free for all”.* Maybe that’s true, but we’ll find out how it all works as well as **some common programming idioms to better encapsulate the internals of objects**.

It’s not necessary to worry about the inner details to be productive. However, most Python coders have a basic awareness of how classes work.

### Dictionary revisited

The Python object system is largely based on an implementation involving dictionaries. They are used for critical parts of the interpreter and may be **the most important type of data in Python**.

For example, a module has `.__dict__` or `globals()`

```py
# foo.py

x = 42
def bar():
    ...

def spam():
    ...

>>> foo.__dict__
{
    'x' : 42,
    'bar' : <function bar>,
    'spam' : <function spam>
}
```

An object has `.__dict__` as well. **In fact, the entire object system is mostly an extra layer that’s put on top of dictionaries.**

```py
>>> s = Stock('GOOG', 100, 490.1)
>>> s.__dict__
{'name' : 'GOOG', 'shares' : 100, 'price': 490.1 }
```

You populate this dict (and instance) when assigning to `self`.

```py
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price
```

**Each instance gets its own private dictionary.**

```py
s = Stock('GOOG', 100, 490.1)
# s.__dict__: {'name' : 'GOOG','shares' : 100, 'price': 490.1 }
t = Stock('AAPL', 50, 123.45)
# t.__dict__: {'name' : 'AAPL','shares' : 50, 'price': 123.45 }
```

If you created 100 instances of some class, there are 100 dictionaries sitting around holding data.

**A separate dictionary** for **class members**, `Stock.__dict__` also holds the methods:

```py
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

    def cost(self):
        return self.shares * self.price

    def sell(self, nshares):
        self.shares -= nshares

>>> Stock.__dict__
{
    'cost': <function>,
    'sell': <function>,
    '__init__': <function>
}
```

Instances and classes are linked together. The `__class__` attribute refers back to the class.

**The instance dictionary holds data unique to each instance, whereas the class dictionary holds data collectively shared by all instances.**

When you work with an object, you access data and methods using the `.` operator.

```py
x = obj.name          # Getting
obj.name = value      # Setting
del obj.name          # Deleting
```

These operations are directly tied to the dictionaries sitting underneath the covers. **Operations that modify an object update the underlying dictionary.**

#### Reading attribute

Suppose you read an attribute on an instance.

```py
x = obj.name
```

The attribute may exist in two places:

- Local instance dictionary.
- Class dictionary.

Both dictionaries must be checked. First, check in local `__dict__`. If not found, look in `__dict__` of class through `__class__`. This lookup scheme is how **the members of a class get shared by all instances**.

#### How inheritance works

Classes may inherit from other classes. The base classes are stored in a tuple in each class. This provides a link to parent classes.

```py
class A(B, C):
    ...

>>> A.__bases__
(<class '__main__.B'>, <class '__main__.C'>)
```

#### Reading Attributes with Inheritance

Logically, the process of finding an attribute is as follows. First, check in local `__dict__`. If not found, look in `__dict__` of the class. If not found in class, look in the base classes through `__bases__`. However, there are some subtle aspects of this discussed next.

#### Reading Attributes with Single Inheritance

In inheritance hierarchies, attributes are found by walking up the inheritance tree in order.

```py
class A: pass
class B(A): pass
class C(A): pass
class D(B): pass
class E(D): pass
```

With single inheritance, there is single path to the top. You stop with the first match.

#### Method Resolution Order (MRO)

Python precomputes an inheritance chain and stores it in the MRO attribute on the class. You can view it.

```py
>>> E.__mro__
(<class '__main__.E'>, <class '__main__.D'>,
 <class '__main__.B'>, <class '__main__.A'>,
 <type 'object'>)
```

This chain is called the **Method Resolution Order**. To find an attribute, Python walks the MRO in order. The first match wins.

#### MRO in Multiple Inheritance

With multiple inheritance, there is no single path to the top. Let’s take a look at an example.

```py
class A: pass
class B: pass
class C(A, B): pass
class D(B): pass
class E(C, D): pass
```

What happens when you access an attribute? An attribute search process is carried out, but what is the order? That’s a problem.

Python uses cooperative multiple inheritance which obeys some rules about class ordering.

- Children are always checked before parents
- Parents (if multiple) are always checked in the order listed.

The MRO is computed by sorting all of the classes in a hierarchy according to those rules.

```py
>>> E.__mro__
(
  <class 'E'>,
  <class 'C'>,
  <class 'A'>,
  <class 'D'>,
  <class 'B'>,
  <class 'object'>)
```

The underlying algorithm is called the [C3 Linearization Algorithm](https://en.wikipedia.org/wiki/C3_linearization). **The gist of the order is: children first, followed by parents.**

#### Why use multiple inheritance: the "Mixin" pattern

Consider two completely unrelated objects:

```py
class Dog:
    def noise(self):
        return 'Bark'

    def chase(self):
        return 'Chasing!'

class LoudDog(Dog):
    def noise(self):
        # Code commonality with LoudBike (below)
        return super().noise().upper()

# And

class Bike:
    def noise(self):
        return 'On Your Left'

    def pedal(self):
        return 'Pedaling!'

class LoudBike(Bike):
    def noise(self):
        # Code commonality with LoudDog (above)
        return super().noise().upper()
```

There is a code commonality in the implementation of `LoudDog.noise()` and `LoudBike.noise()`. In fact, the code is exactly the same. *Naturally, code like that is bound to attract software engineers.*

The **Mixin pattern** is a class with a fragment of code.

```py
class Loud:
    def noise(self):
        return super().noise().upper()
```

This class is not usable in isolation. It mixes with other classes via inheritance.

```py
class LoudDog(Loud, Dog):
    pass

class LoudBike(Loud, Bike):
    pass
```

Miraculously, loudness was now implemented just once and reused in two completely unrelated classes. **This sort of trick is one of the primary uses of multiple inheritance in Python.**

#### Why `super()`

Always use `super()` when overriding methods. `super()` delegates to the next class on the MRO.

The tricky bit is that you don’t know what it is. You especially don’t know what it is if multiple inheritance is being used.

**Caution: Multiple inheritance is a powerful tool. Remember that with power comes responsibility. Frameworks / libraries sometimes use it for advanced features involving composition of components.**

### Encapsulation techniques

When writing classes, it is common to try and encapsulate internal details. This section introduces a few Python programming idioms for this including private variables and properties.

#### Public vs. private

One of the primary roles of a class is to **encapsulate data and internal implementation details of an object**. However, a class also **defines a public interface that the outside world is supposed to use to manipulate the object**. This distinction between implementation details and the public interface is important.

However, in Python, almost everything about classes and objects is open.

- You can easily inspect object internals.
- You can change things at will.
- There is no strong notion of access-control (i.e., private class members)

That is an issue when you are trying to isolate details of the internal implementation.

Python relies on **programming conventions** to indicate the intended use of something. These conventions are based on **naming**. There is a general attitude that it is up to the programmer to observe the rules as opposed to having the language enforce them.

Any attribute name with leading `_` is considered to be **private**.

```py
class Person(object):
    def __init__(self, name):
        self._name = 0
```

You *can* still modify it, it's just a naming style. If you find yourself using such names directly, you’re probably doing something wrong. Look for higher level functionality.

#### `@property`: applying property checks

If you want to enforce some checks on the properties:

```py
s.shares = '50'     # Raise a TypeError, this is a string
```

Use the `@property` decorator.

```py
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

    @property
    def shares(self):
        return self._shares

    @shares.setter
    def shares(self, value):
        if not isinstance(value, int):
            raise TypeError('Expected int')
        self._shares = value

"""
Normal attribute access now triggers the getter and setter methods
under @property and @shares.setter.
"""
>>> s = Stock('IBM', 50, 91.1)
>>> s.shares         # Triggers @property
50
>>> s.shares = 75    # Triggers @shares.setter
```

With this pattern, there are no changes needed to the source code. The new setter is also called when there is an assignment within the class, including inside the `__init__()` method.

```py
class Stock:
    def __init__(self, name, shares, price):
        ...
        # This assignment calls the setter below
        self.shares = shares
        ...

    ...
    @shares.setter
    def shares(self, value):
        if not isinstance(value, int):
            raise TypeError('Expected int')
        self._shares = value
```

**There is often a confusion between a property and the use of private names. Although a property internally uses a private name like `_shares`, the rest of the class (not the property) can continue to use a name like `shares`.**

#### `@property`: uniform access

Properties are also useful for computed data attributes.

```py
class Stock:
    def __init__(self, name, shares, price):
        self.name = name
        self.shares = shares
        self.price = price

    @property
    def cost(self):
        return self.shares * self.price
    ...

>>> s = Stock('GOOG', 100, 490.1)
>>> s.shares # Instance variable
100

"""
Notice, s.cost does not need ()
"""
>>> s.cost   # Computed Value
49010.0
```

**`@property` on a method allows you to *drop the parentheses*, hiding the fact that it’s actually a method!**

It makes things look more uniform for methods that look like data attributes. It is called "uniform access".

```py
>>> s = Stock('GOOG', 100, 490.1)
>>> a = s.cost() # Method
49010.0
>>> b = s.shares # Data attribute
100
```

#### `__slot__` attribute

You can restrict the set of attributes names. It will raise an error for other attributes.

```py
class Stock:
    __slots__ = ('name','_shares','price')
    def __init__(self, name, shares, price):
        self.name = name
        ...

>>> s.price = 385.15
>>> s.prices = 410.2
Traceback (most recent call last):
File "<stdin>", line 1, in ?
AttributeError: 'Stock' object has no attribute 'prices'
```

Although this prevents errors and restricts usage of objects, **it’s actually used for performance and makes Python use memory more efficiently.**

It should be noted that __slots__ is most commonly used as an optimization on classes that serve as data structures. Using slots will make such programs use far-less memory and run a bit faster. You should probably avoid __slots__ on most other classes however.

#### A comment

Don’t go overboard with private attributes, properties, slots, etc. They serve a specific purpose and you may see them when reading other Python code. However, they are not necessary for most day-to-day coding.

### `__init__` vs `__call__`

`__init__` vs `__call__` for a class: https://stackoverflow.com/questions/9663562/what-is-the-difference-between-init-and-call

`__init__` uses the class name and creates an instance.

`__call__` is for an instance to be called as a function.

```py
inst = MyClass()  # MyClass.__init__
s = inst()  # MyClass.__call__
```

## Reference

<https://dabeaz-course.github.io/practical-python/Notes>
