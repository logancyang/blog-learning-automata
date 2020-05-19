---
toc: true
layout: post
description: Quick reference for Python tricks
categories: [note, python]
title: "Handy Python Notes"
---

### Key Concepts

* Immutable = hashable = can be element of a set (or key of dict)
* mutable = unhashable = cannot be element of a set (or key of dict)

* immutables in Python:
  * int, float, long, complex
  * str
  * tuple
  * bytes
  * frozenset

* mutables
  * byte array
  * list
  * set
  * dict

* Do not modify an container while looping over it. Create a copy and loop on that.

* DO NOT ASSIGN STRING CHAR AS IN LIST!!!
  * list(string), modify, and ‘’.join(list) to get string back
  * To sort a string:

{% highlight python %}
>>> s = "dcba"
>>> sorted_s_list = sorted(s)
['a', 'b', 'c', 'd']
>>> sorted_s = ''.join(sorted(s))
"abcd"
{% endhighlight %}

* An example of function:

{% highlight python %}
S = [9, 9, 8, 7, 6]
# cannot swap S items without passing S in, if a and b are indices
def swapWrong(a, b):    X
    tmp = a
    a = b
    b = tmp

>>> swapWrong(S[0], S[3])
>>> S
[9, 9, 8, 7, 6]

def swap(S, a, b):        V
    tmp = S[a]
    S[b] = S[a]
    S[a] = tmp

>>> swap(S, 0, 3)
>>> S
[7, 9, 8, 9, 6]
{% endhighlight %}

<br>

### Swap 2 Elements in a List

* Usually in all languages it looks like,

{% highlight python %}
tmp = A[i]
A[i] = A[j]
A[j] = tmp
{% endhighlight %}

* In Python it can be as simple as `A[i], A[j] = A[j], A[i]`!


<br>

### The List Count Method

* The `list` data structure in Python has a method `list.count(element)` which takes O(n) time.
* Can be high dimensional list, e.g. `board[:m][:n].count(element)`.
  * Side note: `set(2DList) => 1DSet`
* If we want all counts, using a `dict` as a histogram gives all counts in one pass.


<br>

### Use Dict as a Histogram

* Don’t forget to check if key exists.

{% highlight python %}
for item in list:
    if item not in map:
        map[item] = 0
    map[item] += 1
{% endhighlight %}

* A better way, use dict.get(key, default=None).
  * **default** -- This is the Value to be returned in case key does not exist.

{% highlight python %}
d = {}
for color in colors:
    d[color] = d.get(color, 0) + 1
{% endhighlight %}

<br>

### Zip 2 Lists into a Dictionary

* zip two lists into a list of tuples
* dict the list of tuples into a dictionary: note, the tuples must have length 2, error if 3
* an easy way of constructing a reversed dictionary

{% highlight python %}
>>> a = [1, 2, 3]
>>> b = ['x', 'y', 'z']
>>> ab = zip(a, b)
>>> ab
[(1, 'x'), (2, 'y'), (3, 'z')]
>>> dict_ab = dict(ab)
>>> dict_ab
{1: 'x', 2: 'y', 3: 'z'}
>>> dict_ba = {value: key for key, value in dict_ab.iteritems()}
>>> dict_ba
{'y': 2, 'x': 1, 'z': 3}
{% endhighlight %}

<br>

### Check if a Float is a Whole Number

* Two methods: `(my_float).is_integer()`, or `my_float % 1 == 0`
* The second one works because in Python, `5.5 % 1 = 0.5`, `5.0 % 1 = 0.0`, and `0.0 == 0` is `True`.

<br>

### Get the Index of the Min/Max Value in a List S

* If there are duplicates, these methods return the first occurrence

{% highlight python %}
#   print max(enumerate(S))    # returns a tuple (ind, val), key is ind 
print max(enumerate(S), key = lambda S: S[1])    # faster 
print S.index(max(S))    # easier to read
{% endhighlight %}

<br>

### Infinity

* `float('inf')` is positive inf, `float('-inf')` is negative
* Be careful not to multiply inf by 0, it yields nan


<br>

### Import .py File as Module

* In Python, modeling a clustering as a set of sets is impossible since **the elements of a set must be immutable**.
* Import self-defined module: put the file.py file into the same directory, and do
* import *module* (as *alias*)


<br>

### Closure

* Closure: <http://en.wikipedia.org/wiki/Closure_%28computer_programming%29>
* Pass function as parameter into another function.

{% highlight python %}
def function_a():
    def function_b():
{% endhighlight %}

<br>

### Lambda

* runtime function with no name
* often used in conjunction with filter(), map(), reduce()

{% highlight python %}
def f (x): return x**2
...
>>> print f(8)
64
>>>
>>> g = lambda x: x**2
>>> print g(8)
64
{% endhighlight %}

* Or pass a small function as argument

{% highlight python %}
>>> pairs = [(1, 'one'), (2, 'two'), (3, 'three'), (4, 'four')]
>>> pairs.sort(key=lambda pair: pair[1])
>>> pairs
[(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
{% endhighlight %}

<br>

### Initialize a 2D Array (Nested List)

* **USE LIST COMPREHENSION!**
* DO NOT USE `[[None] * ncol] * nrow]`!  `[anything] * const` has sublists pointing to the same object.

{% highlight python %}
S = [[0 for ncol in xrange(5)] for nrow in xrange(3)]
>>> S
[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
{% endhighlight %}

* Similarly, to initialize a triangle where ncol <= nrow:
  * don’t forget the nrow+1

{% highlight python %}
S = [[0 for ncol in xrange(nrow+1)] for nrow in xrange(3)]
>>> S
[[0], [0, 0], [0, 0, 0]]
{% endhighlight %}

* To initialize a contant 2D(or even higher dimensional) list with the shape of a given list "tri"

{% highlight python %}
tri = [
     [2],
    [3,4],
   [6,5,7],
  [4,1,8,3]
]

>>> f = [[0 for item in tri[nrow]] for nrow in xrange(len(tri))]
[[0], [0, 0], [0, 0, 0], [0, 0, 0, 0]]
{% endhighlight %}

* To initialize a 2D list f1 where the value of the first row is 0 to j and the first col is 0 to i.
The commas and colons should be removed, just to make it more readable.

{% highlight python %}
>>> f1 = [[j if i == 0, else: i if j == 0, else: 0 for j in xrange(4)] for i in xrange(4)]
[[0, 1, 2, 3], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
{% endhighlight %}

<br>

### Flatten a 2D Array

* sublist in matrix, then, item in sublist.
* **Nested loops in List Comprehension: outer loop first, then inner loop.**
* flatten = [item for sublist in matrix for item in sublist]

{% highlight python %}
# equivalent to
flatten = []
for sublist in matrix:
    for item in sublist:
        flatten.append(item)
{% endhighlight %}

<br>

### Find Min/Max in 2D Nested List

{% highlight python %}
>>> S = [[random.randint(1, 10) for ncol in xrange(nrow+1)] for nrow in xrange(3)]
>>> S
[[3], [1, 2], [5, 9, 2]]
>>> min(min(sublist) for sublist in S)
1
{% endhighlight %}

<br>

### To Check a List of Booleans to Yield a Boolean: all(), any()

{% highlight python %}
list S = [True, True, True]
>>> all(S) is True
True
>>> any(S) is False
False
{% endhighlight %}

* Beware, DO NOT use it like `all([2, 2, 2]) == 2`, `all()` and `any()` are for booleans inside.

<br>

### If-Else in List Comprehension

* To initialize a list f from list S, if S[i] == 0, f[i] = 1, else f[i] = 0. Here is a 2D example:

{% highlight python %}
>>> f = [[1 if S[i][j] == 0 else 0 for j in xrange(len(S[0]))] for i in xrange(len(S))]
{% endhighlight %}

* [expr **if** S[] == val **else** *expr* **for** i **in** iterable]

* What about elif??

{% highlight python %}
l = [1, 2, 3, 4, 5]

for values in l:
    if values==1:
        print 'yes'
    elif values==2:
        print 'no'
    else:
        print 'idle'

>>> ['yes' if v == 1 else 'no' if v == 2 else 'idle' for v in l]
['yes', 'no', 'idle', 'idle', 'idle']
{% endhighlight %}

* To initialize a 2D (4*4) list f where `f[i][0] = i, f[0][j] = j, else 0`:

{% highlight python %}
>>> [[j if i == 0 else i if j == 0 else 0 for j in xrange(4)] for i in xrange(4)]
[[0, 1, 2, 3], [1, 0, 0, 0], [2, 0, 0, 0], [3, 0, 0, 0]]
{% endhighlight %}

* [expr **if** i == val **else** *expr* **if** j == val **else** *expr* **for** i **in** iterable]
* very powerful pythonic list comprehension!!

<br>

### Create Dict with Keys from a Set or List

{% highlight python %}
# create a dict with keys from a set (or list), value 0 
T = set([1, 2, 4])
 dictT = dict((element, 0) for element in T) 
print dictT

# output:
{1: 0, 2: 0, 4: 0}
{% endhighlight %}

<br>

### Using Lists as Stacks

* It's very easy to use lists as stacks with append() and pop(). Last in first out.

{% highlight python %}
>>> stack = [3, 4, 5]
>>> stack.append(6)
>>> stack.append(7)
>>> stack
[3, 4, 5, 6, 7]
>>> stack.pop()
7
>>> stack
[3, 4, 5, 6]
>>> stack.pop()
6
>>> stack.pop()
5
>>> stack
[3, 4]
{% endhighlight %}

<br>

### Using Lists as Queues

* We can use pop(0) and append() to implement the queue behavior but it's not efficient.
Because popping from the first position is slow, all elements must shift. Instead it's better to use
collections.deque and its append() and popleft() methods.

{% highlight python %}
>>> from collections import deque
>>> queue = deque(["Eric", "John", "Michael"])
>>> queue.append("Terry")           # Terry arrives
>>> queue.append("Graham")          # Graham arrives
>>> queue.popleft()                 # The first to arrive now leaves
'Eric'
>>> queue.popleft()                 # The second to arrive now leaves
'John'
>>> queue                           # Remaining queue in order of arrival
deque(['Michael', 'Terry', 'Graham'])
{% endhighlight %}

<br>

### A Peculiar Usage of Augment Assignment in Python

* The augmented addition operator `+=` behaves unexpectedly in the following case,

{% highlight python %}
>>> lst = []
>>> print lst + "123"
TypeError: can only concatenate list (not "str") to list
>>> lst += "123"
>>> print lst
['1', '2', '3']
{% endhighlight %}

* What happened is that `+=` calls __iadd__() method and try to modify the list in-place,
while adding all the elements of the iterable on the right to the list.

<br>

### Class Method vs. Static Method
* `@staticmethod` function is nothing more than a function defined inside a class.
It is callable without instantiating the class first. It’s definition is immutable via inheritance.

* `@classmethod` function also callable without instantiating the class,
but its definition follows Sub class, not Parent class, via inheritance. That’s because the first argument for
`@classmethod` function must always be cls (class).

*To be continued...*

<br>

#### Some good resources:

* [The tutorial from Python documentation](https://docs.python.org/2/tutorial/)
* [Data Structures](https://docs.python.org/2/tutorial/datastructures.html#)
* [I/O](https://docs.python.org/2/tutorial/inputoutput.html)
* [Errors and Exceptions](https://docs.python.org/2/tutorial/errors.html)
* [Classes, Iterators and Generators](http://anandology.com/python-practice-book/iterators.html)
* [Standard Library](https://docs.python.org/2/library/)

