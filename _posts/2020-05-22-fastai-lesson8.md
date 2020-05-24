---
toc: true
layout: post
description: fast.ai note series
categories: [note, fastai]
title: "FastAI Lesson 8: Backprop from the Foundations"
---

## Jeremy's starting comments

- "cutting-edge deep learning" now is more and more about engineering and not papers.
It's about **who can make things in code that work properly**.
- Part II of fastai is bottom-up learning **with code**. It helps you understand the connections between algorithms,
and make your own algorithm for your own problem, and debug, profile, maintain it.
- Swift and Julia are the promising languages for high performance computing.

### Swift for TensorFlow vs. PyTorch

<img src="{{ site.baseurl }}/images/fastai/s4tf.png" alt="S4TF" align="middle"/>

Swift is a thin layer on top of LLVM. LLVM compiles Swift code to super fast machine code.

Python is the opposite. We write Python as an interface but things usually run in C++. It prevents doing deep dives
as we shall see in this course.

*Opportunity: join the Swift for TF community to contribute and be a pioneer in this field!*

### Recreate `fastai` Library from Scratch

<img src="{{ site.baseurl }}/images/fastai/create_fastai.png" alt="fastai" align="middle"/>

Benefit of doing this

- *Really* experiment
- Understand it by creating it
- Tweak everything
- **Contribute**
- Correlate papers with code

*Opportunities*

- Make homework at the cutting edge
- There are few DL practitioners that know what you know now
- Experiment lots, especially in your area of expertise
- Much of what you find will have not be written about before
- Don't wait to be perfect before you start communicating. Write stuff down for the person you were 6 months ago.

### How to Train a Good Model

<img src="{{ site.baseurl }}/images/fastai/trainsteps.png" alt="fastai" align="middle"/>

5 steps of reducing overfitting
- more data
- data augmentation
- generalization architectures
- regularization
- reducing architecture complexity *(this should be the last step)*

### Start Reading Papers

Get pass the fear of Greek letters! It's just code.

*Opportunity: there are blog posts that describing a paper better than the paper does. Write these blog posts!*

Read blog posts and also the paper itself.

## Goal: Recreating a Modern CNN Model

<img src="{{ site.baseurl }}/images/fastai/recreatecnn.png" alt="fastai" align="middle"/>

For development, Jeremy recommends `nbdev` for library development in Jupyter notebook.

Tip: Python's `fire` library lets you convert a function into CLI.

notebook: `01_matmul`

Important: get familiar with PyTorch Tensors. It can do everything like a numpy array and it can run on GPU.

`tensor.view()` is equivalent to `nparray.reshape()`.

## Creating Matrix Multiplication

### Pure Python with 3 nested loops (speed lvl0)

Implement matrix multiplaction with 3 loops.

```py
def matmul(a,b):
    ar,ac = a.shape # n_rows * n_cols
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            for k in range(ac): # or br
                c[i,j] += a[i,k] * b[k,j]
    return c
```

This is super slow because it's in Python. A `(5, 784) by (784, 10)` matrix multiplication took ~1s. MNIST needs ~10K of them, so it will take 10K seconds, that's unacceptable.

The way to speed this up is to use something other than Python -- use PyTorch where it uses `ATen` (C++) under the hood.

Tip: to get LaTeX formula, go to Wikipedia and click edit. Or go to Arxiv and do `Download other format` on the top right, then `download source`.

### Elementwise vector operations with 2 nested loops (speed lvl1)

```py
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
        for j in range(bc):
            # Any trailing ",:" can be removed
            # a[i, :] means the whole ith row
            # b[:, j] means the whole jth col
            # This is not really Python, it tells Python to call C
            c[i,j] = (a[i,:] * b[:,j]).sum()
    return c
```

This is hundreds of times faster. *Next, broadcasting makes it even faster.*

Tip: to test equal for floats, set a tolerance and use something like `torch.allclose()`. Float's implementation gives
small numerical errors.

### Broadcasting with 1 loop (speed lvl2)

Broadcasting is the most powerful tool to speed things up. It gets rid of for loops and does implicit broadcast loops.

Any time we use broadcasting, we are using C speed (on CPU) or CUDA speed (on GPU).

Easiest example is `a + 1` where `a` is a tensor. `1` is automatically turned into a tensor that matches the shape of `a`. This is scalar to tensor.

We can also broadcast vector to higher order tensors.

```py
c = tensor([10.,20,30])
m = tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
c + m
"""
tensor([[11., 22., 33.],
        [14., 25., 36.],
        [17., 28., 39.]])

We don't really copy the rows, the rows are given a *stride* of 1.
"""
# To check what c looks like after broadcasting
t = c.expand_as(m); t
"""
tensor([[10., 20., 30.],
        [10., 20., 30.],
        [10., 20., 30.]])

Use t.storage() we can check the memory usage!
It shows we only have one row in memory, not really making a full matrix
during broadcasting.

Use t.stride() shows
(0, 1)
meaning stride is 0 for rows, 1 for columns.

This idea is used in all linear algebra libraries.
"""
```

**To add a dimension, use `unsqueeze(axis)`, or, use `None` at that axis when indexing**. Example:

```py
# These are not in-place, c is not updated
c.unsqueeze(0)
# or
c[None, :]
"""
tensor([[10., 20., 30.]])
"""

c.unsqueeze(1)
# or
c[:, None]
"""
tensor([[10.],
        [20.],
        [30.]])
"""
```

Tip: always use `None` over `unsqueeze` because it's more convenient and we can add more than one axis.

Trick:
- We can omit trailing `,:` as in `c[None, :] == c[None]`
- We can use `...` as in `c[:, None] == c[..., None]`. This is helpful especially when we don't know the rank of the tensor.

```py
def matmul(a,b):
    ar,ac = a.shape
    br,bc = b.shape
    assert ac==br
    c = torch.zeros(ar, bc)
    for i in range(ar):
#       c[i,j] = (a[i,:]          * b[:,j]).sum() # previous
        # Notice we got rid of loop j, that's why it's even faster
        # than the elementwise ops previously
        c[i]   = (a[i].unsqueeze(-1) * b).sum(dim=0)
        # Or c[i] = (a[i][:, None] * b).sum(dim=0)
    return c
```

Concrete example for the above code:

```py
a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
])

b = torch.tensor([
    [10, 10],
    [20, 20],
    [30, 30]
])

# For i = 0
a0 = a[0][:, None]
"""
a0:
tensor([[1],
        [2],
        [3]])

b:
tensor([[10, 10],
        [20, 20],
        [30, 30]])

a0 * b: we rotate a0, a row vector to be a col vector,
and broadcast into shape of b, and does elementwise *

tensor([[10, 10],
        [40, 40],
        [90, 90]])

Then sum over axis=0 (rows)

tensor([140, 140])

This is the result of matmul for a row vector a[0] and matrix b.

Do the same for the 2nd row of a, we have a @ b:

tensor([[140, 140],
        [320, 320]])
"""
```

Now we only have one level of loop for the matrix multiplication, and it's 1000x times faster than the raw Python version of 3 nested loops.

### Broadcasting Rule

```py
c = tensor([10., 20., 30.])

# Add leading axis 0, row
c[None,:], c[None,:].shape
"""
tensor([[10., 20., 30.]]), torch.Size([1, 3])
"""

# Add trailing axis, col
c[:,None], c[:,None].shape
"""
tensor([[10.],
        [20.],
        [30.]])
torch.Size([3, 1])
"""

# How does this do broadcasting?
# Here is where the BROADCASTING RULE comes in
# Where there's a missing dimension, np/pytorch fills in a dimension
# with size 1. A dim of size 1 can be broadcast into any size.
# E.g. (1, 1, 3) * (256, 256, 3) -> (256, 256, 3)
c[None,:] * c[:,None]
"""
tensor([[100., 200., 300.],
        [200., 400., 600.],
        [300., 600., 900.]])
"""

# Similarly
c[None] > c[:,None]
"""
tensor([[0, 1, 1],
        [0, 0, 1],
        [0, 0, 0]], dtype=torch.uint8)
"""
```

When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when

- they are equal, or
- one of them is 1, in which case that dimension is broadcasted to make it the same size

Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

    Image  (3d array): 256 x 256 x 3
    Scale  (1d array):             3
    Result (3d array): 256 x 256 x 3

The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.

### Einstein Summation with no loops (speed lvl3)

Einstein summation notation: `ik,kj->ij`

`ik,kj`: input

`ij`: output

Each letter is the size of a dimension. *Repeated letters indicate dot product*.

```py
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```

This is even faster. And we can create new operations easily such as batch matrix multiplication: `bik,bkj->bij`.

But having a string as a language inside of a language is not a good idea, e.g. regex. We should be able to write Swift and Julia operating at this speed in a few years.

### PyTorch op with no loops (speed lvl4)

PyTorch's `matmul` or `@` operation is even faster than `einsum`, it's ~50K times faster than raw Python. Because to do really fast matrix multiplication on big matrices, it can't fit in CPU cache and needs to be chopped down into smaller matrices. `BLAS` libraries do that. Examples are NVidia's `cuBLAS`, and Intel's `mkl`.

## Fully Connected Nets

### Forward pass

First, load the data and *apply normalization*.

Note: use the training set's mean and std to normalize the validation set! Always make sure the validation set and the training set are normalized in the same way.

```py
# num hidden
nh = 50
# simplified kaiming init / he init: divide by sqrt(n_inputs). m is # examples
w1 = torch.randn(m,nh)/math.sqrt(m)
b1 = torch.zeros(nh)
w2 = torch.randn(nh,1)/math.sqrt(nh)
b2 = torch.zeros(1)

# This should be ~ (0,1) (mean,std)...
x_valid.mean(),x_valid.std()

def lin(x, w, b):
    """Linear layer"""
    return x@w + b

t = lin(x_valid, w1, b1)
# The effect of Kaiming init: makes the linear output have
# ~0 mean and 1 std
t.mean(),t.std()
```

`Kaiming init` is a very important factor to train deep networks. Some researchers trained a 10K-layer network without normalization layers just with *careful initialization*.

```py
def relu(x): return x.clamp_min(0.)
```

Tip: if there's a function for some calculation in pytorch, such as `clamp_min`, it's generally written in C and it's faster than your implementation in Python.

### Kaiming Init

From pytorch docs: `a: the negative slope of the rectifier used after this layer (0 for ReLU by default)`

$$\text{std} = \sqrt{\frac{2}{(1 + a^2) \times \text{fan\_in}}}$$

This was introduced in the paper that described the Imagenet-winning approach from *He et al*: [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852), which was also the first paper that claimed "super-human performance" on Imagenet (and, most importantly, it introduced resnets!)

Simply put, **for ReLU, if the inputs are mean 0 and std 1, the numbers below 0 are clipped so we lose half the variance. The way to fix it is to time it by 2, proposed in the paper.**

```py
# kaiming init / he init for relu
w1 = torch.randn(m,nh)*math.sqrt(2/m)
w1.mean(),w1.std()
# (tensor(0.0001), tensor(0.0508))

t = relu(lin(x_valid, w1, b1))
t.mean(),t.std()
# (tensor(0.5678), tensor(0.8491))
```

Conv layers can also be looked at as linear layers with a special weight matrix where there are a lot of 0s for the pixels outside the filter, so this initialization does the same thing for them.

In pytorch,

```py
init.kaiming_normal_(w1, mode='fan_out')

# check doc by `init.kaiming_normal_??`
```

Then we can write the model and the loss function. We use MSE for now for simplicity.

```py
def model(xb):
    l1 = lin(xb, w1, b1)
    l2 = relu(l1)
    l3 = lin(l2, w2, b2)
    return l3

def mse(output, targ): return (output.squeeze(-1) - targ).pow(2).mean()
y_train,y_valid = y_train.float(),y_valid.float()
preds = model(x_train)
mse(preds, y_train)
```

### Backward pass

All you need to know about matrix calculus from scratch: <https://explained.ai/matrix-calculus/index.html>

```py
def mse_grad(inp, targ):
    # grad of loss with respect to output of previous layer
    inp.grad = 2. * (inp.squeeze() - targ).unsqueeze(-1) / inp.shape[0]

def relu_grad(inp, out):
    # grad of relu with respect to input activations
    inp.grad = (inp>0).float() * out.grad

def lin_grad(inp, out, w, b):
    # grad of matmul with respect to input
    inp.grad = out.grad @ w.t()
    w.grad = (inp.unsqueeze(-1) * out.grad.unsqueeze(1)).sum(0)
    b.grad = out.grad.sum(0)

def forward_and_backward(inp, targ):
    # forward pass:
    l1 = inp @ w1 + b1
    l2 = relu(l1)
    out = l2 @ w2 + b2
    # we don't actually need the loss in backward!
    # this is just here if we want to print it out!
    loss = mse(out, targ)

    # backward pass:
    mse_grad(out, targ)
    lin_grad(l2, out, w2, b2)
    relu_grad(l1, l2)
    lin_grad(inp, l1, w1, b1)
```

#### Layers as Classes

Observe the above code, we see each function for grad can take in inputs, weight and bias. We can make layer classes to have inputs, weight and bias, and define a `forward()` to calculate outputs, and `backward()` to calculate the gradients. Refactor the previous code,

```py
class Relu():
    # Notice this Relu() class does not have __init__
    # Instantiating an instance is just `Relu()`
    # dunder call means we can use the class name as a function!
    def __call__(self, inp):
        self.inp = inp
        self.out = inp.clamp_min(0.)-0.5
        return self.out

    def backward(self): self.inp.g = (self.inp>0).float() * self.out.g


class Lin():
    def __init__(self, w, b):
        self.w, self.b = w, b

    def __call__(self, inp):
        self.inp = inp
        self.out = inp@self.w + self.b
        return self.out

    def backward(self):
        self.inp.g = self.out.g @ self.w.t()
        # Creating a giant outer product, just to sum it, is inefficient!
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.b.g = self.out.g.sum(0)


class Mse():
    def __call__(self, inp, targ):
        self.inp = inp
        self.targ = targ
        self.out = (inp.squeeze() - targ).pow(2).mean()
        return self.out

    def backward(self):
        self.inp.g = 2. * (self.inp.squeeze() - self.targ).unsqueeze(-1) / self.targ.shape[0]


class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```

#### Layers as Modules

We see that all layers have outputs, forward and backward passes. Further refactoring, we introduce the `Module` class (similar to pytorch `nn.Module`).

```py
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self): raise Exception('not implemented')
    def backward(self): self.bwd(self.out, *self.args)


class Relu(Module):
    def forward(self, inp): return inp.clamp_min(0.) - 0.5
    def bwd(self, out, inp): inp.g = (inp>0).float() * out.g


class Lin(Module):
    def __init__(self, w, b): self.w,self.b = w,b

    def forward(self, inp): return inp@self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g @ self.w.t()
        self.w.g = inp.t() @ out.g
        self.b.g = out.g.sum(0)


class Mse(Module):
    def forward (self, inp, targ):
        return (inp.squeeze() - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2 * (inp.squeeze()-targ).unsqueeze(-1) / targ.shape[0]


class Model():
    def __init__(self):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = Mse()

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for l in reversed(self.layers): l.backward()
```


## Equivalent Code in PyTorch


```py
from torch import nn

class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [
            nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)
        ]
        self.loss = mse

    def __call__(self, x, targ):
        for l in self.layers: x = l(x)
        return self.loss(x.squeeze(), targ)

model = Model(m, nh, 1)

%time loss = model(x_train, y_train)

%time loss.backward()
```

In the next lesson, we will get the training loop, the optimizer, and other loss functions.

## Homework

- Read Kaiming's paper: Delving Deep into Rectifiers. Focus on section 2.2.
- Xavier init paper is also really readable, we will implement a lot from it.

## My Random Thoughts

The career path after fast.ai should be something that mixes engineering and research:

- **Applied scientist** or **research engineer**. It's different from the usual "data scientist", which focuses on analytics, business metrics and non-DL work (lacks in engineering and research in DL); it's also different from ML engineer, which focuses on productionizing and maintaining models (lacks in research).
- Open source contribution to key DL projects. Swift for Tensorflow is one advocated by Jeremy.

Some one who can implement DL frameworks from scratch and grasp key DL research shouldn't be a "data scientist" or "ML engineer" in a non-research organization. There are tons of data scientists and ML engineers out there, but those who can reach high level of fast.ai competence are rare.

**Demonstrate the knowledge by blogging and making a great project.**
