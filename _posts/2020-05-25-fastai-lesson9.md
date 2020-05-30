---
toc: true
layout: post
description: fast.ai note series
categories: [note, fastai]
title: "FastAI Lesson 9: How to Train Your Model"
comments: true
---

Tip: always include this following code as the 1st cell of any notebook to avoid restarting kernel for imported module changes.

```py
%load_ext autoreload
%autoreload 2
```

## Jeremy's starting comments: how to do research and software development

Jeremy shows how he did research into why `sqrt(5)` was used in pytorch's kaiming initialization.

**The question is, does this initialization make `nn.Conv2d` work well.**

Notebook: `02a_why_sqrt5`

Note: `init.kaiming_normal_(weight, a)` is designed to be used after a (leaky) ReLU layer. Here `a` is the "leak" of the leaky ReLU, i.e. the *gradient for the side inputs < 0*.

Glossary: `rec_fs`, or *receptive field size*, is # elements in a convolution kernel. A 5x5 kernel has `rec_fs == 25`.

Going through the notebook, the results show that the variance keeps getting smaller as there are more layers added, which is a concerning issue.

Jeremy reached out to the pytorch team and got a response that it was a historical bug from the original torch implementation. Then they created an [issue](https://github.com/pytorch/pytorch/issues/18182) to fix it.

The moral of the story is that in deep learning, don't assume everything in the library is right. It doesn't take much to go digging up the code and try making sense of it.

If you find a problem, make your research into a gist and share with the community or the team maintaining the library.

Note: notebook `02b_initializing` shows that a series of matrix multiplications can explode or diminish quickly if not properly initialized. Training deep networks require good initializations for this reason, because DNN is essentially a series of matmuls.

*Recommended paper: [All You Need is a Good Init](https://arxiv.org/abs/1511.06422)*

### Fun fact 1

A fun fact is that there is a Twitter handle [@SELUAppendix](https://twitter.com/seluappendix?lang=en) that mocks the fact that [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) had a 96-page appendix for the math it used to get good inits. If you add dropout or any change to the network you'll need to go through that math again.

> twitter: https://twitter.com/SELUAppendix/status/873882218774528003

### Fun fact 2

Another fun fact is that pytorch's linear layer does a transpose first because of historical reasons. We created a linear layer with input dimension 784 and output dimension 50 (hidden layer dimension), so the shape is `(784, 50)`. The pytorch linear layer has shape `(50, 784)` because the old Lua couldn't handle batch matrix multiplication without this transpose.

In this particular case, it doesn't make things slower so it doesn't matter. But in a lot of cases, these things do matter.

## Recreate a modern CNN: the training loop

Notebook: `03_minibatch_training`

### Create the Cross-Entropy Loss Function

First we introduce softmax and negative log-likelihood (cross-entropy loss).

The cross entropy loss for some target $y$ and some prediction $\hat{y}$ is given by:

$$ \text{NLL} = -\sum_{0 \leq i \leq n-1} y_i\, \log \hat{y}_i $$

where

$$\hat{y}_i = \text{softmax}(\mathbf{x})_i = \frac{e^{x_{i}}}{\sum_{0 \leq j \leq n-1} e^{x_{j}}}$$

But since **target $y$s are 1-hot encoded**, this can be rewritten as $-\log(\hat{y}_i)$ where i is the index of the desired target.

---
In the case of binary classification,

$$\text{NLL} = -y \log(\hat{y}) - (1-y) \log(1 - \hat{y})$$

The coefficients before logs are just a way of **selection**, i.e. y = 1 then select the 1st term, y = 0 then select the 2nd term.

---


*Tip: multiplying with a one-hot encoded vector is equivalent to a **selection** where the vector is 1. Don't do the actual multiplication.*

Trick:

```py
def nll(softmax_preds, targets):
    """
    Use array indexing to select the corresponding values for
    cross entropy loss.
    """
    log_sm = softmax_preds.log()
    return -log_sm[range(targets.shape[0]), targets].mean()

# Example:
smpred = torch.Tensor([[.01, .98, .01], [.001, .001, .998]])
#                            ----                    ----
# The negative log of the softmax predictions: very close to 0 at places
# that were close to 1 in the softmax output
# tensor([[4.6052, 2.0203e-02, 4.6052],
#         [6.9078, 6.9078, 2.0020e-03]])
targets = torch.LongTensor([1, 2])
# nll picks out the elements from each of row in smpred with the
# indices in targets
nll(smpred, targets)
# This exxample has very good softmax prediction so the overall
# cross entropy loss is close to 0
# tensor(0.0111)
```

### Numerical Stability Considerations

`exp()` creates huge numbers, it creates big errors in floating point. **To avoid this numerical stability problem, we use the [LogSumExp](https://en.wikipedia.org/wiki/LogSumExp) trick**.

$$\log \left ( \sum_{j=1}^{n} e^{x_{j}} \right ) = \log \left ( e^{a} \sum_{j=1}^{n} e^{x_{j}-a} \right ) = a + \log \left ( \sum_{j=1}^{n} e^{x_{j}-a} \right )$$

where a is the maximum of the $x_{j}$.

In code,

```py
# Avoid overflow caused by huge numbers from exp()
def logsumexp(x):
    m = x.max(-1)[0]
    return m + (x-m[:,None]).exp().sum(-1).log()
```

pytorch also has `logsumexp()`.

Note: in pytorch,

**`F.nll_loss(F.log_softmax(pred, -1), y_train)` is equivalent to `F.cross_entropy(pred, y_train)`.**

Now, we have implemented cross-entropy loss for multiclass classification from scratch.

For accuracy, do this

```py
def accuracy(pred, yb):
    return (torch.argmax(pred, dim=1)==yb).float().mean()
```

Notice that pytorch tensor can only use `mean()` on float type.

### Implement the Training Loop

We need to refactor our `Module` class to be able to get all the model parameters so that we can update them later.

```py
class DummyModule():
    def __init__(self, n_in, nh, n_out):
        self._modules = {}
        self.l1 = nn.Linear(n_in,nh)
        self.l2 = nn.Linear(nh,n_out)

    def __setattr__(self,k,v):
        """
        This is a special Python dunder method. Every time __init__ is
        called, this is called to do something for the attributes.
        """
        # Methods start with _ are internal. Need this condition to
        # avoid infinite recursion
        if not k.startswith("_"): self._modules[k] = v
        # Set attribute for parent, in this case just the Python object
        super().__setattr__(k,v)

    def __repr__(self): return f'{self._modules}'

    def parameters(self):
        """Returns a generator"""
        for l in self._modules.values():
            for p in l.parameters(): yield p
```

Note that `__setattr__(key, value)` is used as a magical method to populate `self._modules` dictionary. `key` turns the attribute variable names into strings. In this case, keys are `l1` and `l2`.

**This is exactly the same as if we inherit from pytorch's `nn.Module`. Pytorch does the `__setattr__` thing to populate the `modules` dictionary for us when we call `super().__init__()` in our Model class.**

Now the training loop is

```py
def fit():
    for epoch in range(epochs):
        for i in range((n-1)//bs + 1):
            start_i = i*bs
            end_i = start_i+bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            loss = loss_func(model(xb), yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): p -= p.grad * lr
                model.zero_grad()
```

#### pytorch nn.ModuleList

With a list of layers we can init a model like this

```py
class SequentialModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        # The line above is equivalent to
        # self.layers = layers
        # for i,l in enumerate(self.layers): self.add_module(f'layer_{i}', l)

    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
```

Note that the `layers` here are objects with forward and backward defined in the previous lesson, so `nn.ModuleList` can work. It doesn't know how to implement forward and backward passes. But `nn.Sequential` does.

#### pytorch nn.Sequential

An even simpler way to init a model is

```py
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

It even takes care of the definition of the forward backward passes.

### Implement the Optimizer Class

To refactor the training loop further to be able to just use

```py
opt.step()
opt.zero_grad()
```

instead of

```py
with torch.no_grad():
    for p in model.parameters(): p -= p.grad * lr
    # For the case of Gradual Unfreezing, the user might want to include
    # only a subset of parameters, so we should avoid model.zero_grad()
    model.zero_grad()
```

We define the Optimizer class,

```py
class Optimizer():
    def __init__(self, params, lr=0.5): self.params,self.lr=list(params),lr

    def step(self):
        """
        This is the purpose of grad computation.
        The update operations doesn't need grad itself.
        """
        with torch.no_grad():
            for p in self.params: p -= p.grad * lr

    def zero_grad(self):
        """
        Only does zero_grad for the parameters passed in, not all model
        parameters in case the user wants gradual unfreezing.
        """
        for p in self.params: p.grad.data.zero_()
```

Jeremy recommends using something like `assert accuracy > 0.7` to make sure the model is doing what it should do after training. It's an indicator whether there's a bug that makes the model wrong.

When developing models, we can embrace randomness by not setting the random seed. We need to see how it works with randomness, which bits are stable and which are not.

For research, in some cases we need reproducibility. We set the seeds in those cases.

## Dataset and Dataloader

### Dataset

With a `Dataset` class we do minibatches easier.

```py
class Dataset():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
assert len(train_ds)==len(x_train)
assert len(valid_ds)==len(x_valid)
```

Now our training loop becomes

```py
for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        """
        # before:
        start_i = i*bs
        end_i = start_i+bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        """
        xb, yb = train_ds[i*bs : i*bs+bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()
```

### Dataloader

Previously, our loop iterated over batches (xb, yb) like this:

```python
for i in range((n-1)//bs + 1):
    xb, yb = train_ds[i*bs : i*bs+bs]
    ...
```

Let's make our loop much cleaner, using a data loader:

```python
for xb, yb in train_dl:
    ...
```

Define the `Dataloader` class that takes a `Dataset` and a batch size and produces the batches for us.

```py
class DataLoader():
    def __init__(self, dataset, bs):
        self.dataset, self.bs = dataset, bs

    def __iter__(self):
        """
        When you call a for loop on something, it calls the __iter__
        behind the scene
        """
        for i in range(0, len(self.dataset), self.bs):
            yield self.dataset[i:i+self.bs]
```

Note: `yield` is a *coroutine* in Python.

TODO: Make note on Python coroutines and AsyncIO.

To use it, write `next(iter(...))`,

```py
xb, yb = next(iter(train_dl))
```

With data loader, our training loop becomes

```py
"""
We now have the cleanest form of a training loop.
One iteration has 5 steps.
"""
def fit():
    for epoch in range(epochs):
        for xb,yb in train_dl:
            # 1. Get predictions
            pred = model(xb)
            # 2. Calculate loss
            loss = loss_func(pred, yb)
            # 3. Calculate gradients
            loss.backward()
            # 4. Update the parameters
            opt.step()
            # 5. Reset the gradients
            opt.zero_grad()
```

This is quite neat and beautiful!

One problem that remains is that we are looping through the data in order. We need to do random sampling to let each batch be different.

### Random Sampling

Define a `Sampler` class

```py
class Sampler():
    def __init__(self, dataset, bs, shuffle=False):
        self.n, self.bs, self.shuffle = len(dataset), bs, shuffle

    def __iter__(self):
        self.idxs = torch.randperm(self.n) if self.shuffle else torch.arange(self.n)
        for i in range(0, self.n, self.bs):
            yield self.idxs[i:i+self.bs]

def collate(b):
    xs,ys = zip(*b)
    return torch.stack(xs),torch.stack(ys)

class DataLoader():
    def __init__(self, dataset, sampler, collate_fn=collate):
        self.dataset, self.sampler, self.collate_fn = dataset,sampler, collate_fn

    def __iter__(self):
        for s in self.sampler: yield self.collate_fn([self.dataset[i] for i in s])


train_samp = Sampler(train_ds, bs, shuffle=True)
valid_samp = Sampler(valid_ds, bs, shuffle=False)

train_dl = DataLoader(train_ds, sampler=train_samp, collate_fn=collate)
valid_dl = DataLoader(valid_ds, sampler=valid_samp, collate_fn=collate)
xb,yb = next(iter(valid_dl))
plt.imshow(xb[0].view(28,28))
yb[0]
```

### Pytorch's Dataloader

```py
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

train_dl = DataLoader(
    train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate)
valid_dl = DataLoader(
    valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate)

# Or omit the sampler and collate function, the ones we implemented are
# the default
train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True)
valid_dl = DataLoader(valid_ds, bs, shuffle=False)
```

### Validation

In pytorch, `model` has a `training` attribute which is boolean.

Take this fitting loop for example, `model.training` is set by `model.train()` and `model.eval()`.

```py
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Handle batchnorm / dropout
        model.train()
#         print(model.training) -> True
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
#         print(model.training) -> False
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in valid_dl:
                pred = model(xb)
                tot_loss += loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv
```

This is useful because for some layers such as batch norm and dropout, they should do their thing in training but they are different during evaluation. This makes sure of that.

Also notice that the loss accumulation in the above code only works when batch sizes are equal. With varying batch sizes, we need weighted average.

### Question: why `zero_grad()` in every iteration?

Answer:

1. We do batch gradient descent and it works by accumulating gradients in each batch. We would want to be able to stitch different components together for the gradients by not calling `zero_grad()` in some cases, so we make it a seperate method.

2. Having a separate `zero_grad()` in the Optimizer class, rather than something like the code below where we zero out the gradients after each step, enables us to *accumulate gradients*.

```py
class Optimizer():
    def __init__(self, params, lr=0.5): self.params,self.lr=list(params),lr

    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= p.grad * lr
                p.grad.data.zero_()
```

For example, if we have big images to train with and can only fit a smaller number in the GPU, we can do

```py
for epoch in range(epochs):
    for i in range((n-1)//bs + 1):
        xb,yb = train_ds[i*bs : i*bs+bs]
        pred = model(xb)
        loss = loss_func(pred, yb)
        loss.backward()

        # THIS EFFECTIVELY DOUBLED OUR BATCH SIZE!
        if i % 2:
            opt.step()
            opt.zero_grad()
```

Of course, we can have better API design by adding `auto_zero` into the Optimizer, e.g.

```py
class Optimizer():
    def __init__(self, params, lr=0.5, auto_zero=True):
        self.params, self.lr, self.auto_zero = list(params), lr, auto_zero

    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= p.grad * lr
            if self.auto_zero: self.zero_grad()

    def zero_grad(self):
        for p in self.params: p.grad.data.zero_()
```

This removes the need to call `zero_grad()` in every batch iteration, which could potentially avoid bugs. But this is not something pytorch has done.

## Callbacks

fast.ai docs on [callbacks](https://docs.fast.ai/callbacks.html)

Notebook: `04_callbacks`

To recap, the training loop we implemented is

<img src="{{ site.baseurl }}/images/fastai/training_loop.png" alt="train_loop" align="middle"/>

<img src="{{ site.baseurl }}/images/fastai/train_loop_picture.png" alt="train_loop_picture" align="middle"/>

Different kinds of models have different training loops. It's intractable to write each type of training loop and it's bad code design. A better way is to insert callbacks at the right events.

<img src="{{ site.baseurl }}/images/fastai/train_loop_callback.png" alt="train_loop_callback" align="middle"/>

<img src="{{ site.baseurl }}/images/fastai/callback_in_code.png" alt="callback_in_code" align="middle"/>

Here are some other callback examples in fastai.

<img src="{{ site.baseurl }}/images/fastai/fastai_callbacks.png" alt="fastai_callbacks" align="middle"/>

This is the callbacks for a GAN training loop,

<img src="{{ site.baseurl }}/images/fastai/callback_gan.png" alt="callback_gan" align="middle"/>

### Refactoring `fit()`

We start by refactoring the `fit()` function.

```py
# Before
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# We get nervous when a function takes in too many parameters
# Need to group relevant ones together.
# E.g. the data loaders can be grouped together first into `DataBunch`
class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset

def get_model(data, lr=0.5, nh=50):
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(
        nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,data.c)
    )
    return model, optim.SGD(model.parameters(), lr=lr)

class Learner():
    # Notice the Learner class has no logic at all
    # It's just a useful device for storing things
    def __init__(self, model, opt, loss_func, data):
        self.model, self.opt, self.loss_func, self.data = \
            model, opt, loss_func, data


x_train, y_train, x_valid, y_valid = get_data()
train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
nh, bs = 50, 64
c = y_train.max().item() + 1
loss_func = F.cross_entropy
data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
learn = Learner(*get_model(data), loss_func, data)

# After
fit(epochs, learn)
```

Note: Python `@property` decorator helps create a getter method so that the property can be accessed by object dot the function name. For more info about it, check [here](https://www.freecodecamp.org/news/python-property-decorator/).

Inside the `fit()` function, `model` becomes `learn.model`, `data` becomes `learn.data`.

```py
def fit(epochs, learn):
    for epoch in range(epochs):
        learn.model.train()
        for xb,yb in learn.data.train_dl:
            loss = learn.loss_func(learn.model(xb), yb)
            loss.backward()
            learn.opt.step()
            learn.opt.zero_grad()

        learn.model.eval()
        with torch.no_grad():
            tot_loss,tot_acc = 0.,0.
            for xb,yb in learn.data.valid_dl:
                pred = learn.model(xb)
                tot_loss += learn.loss_func(pred, yb)
                tot_acc  += accuracy (pred,yb)
        nv = len(learn.data.valid_dl)
        print(epoch, tot_loss/nv, tot_acc/nv)
    return tot_loss/nv, tot_acc/nv

loss, acc = fit(1, learn)
```

### Add Callbacks

Implement the `Callback` class,

```py
class Callback():
    def begin_fit(self, learn):
        self.learn = learn
        return True
    def after_fit(self): return True
    def begin_epoch(self, epoch):
        self.epoch=epoch
        return True
    def begin_validate(self): return True
    def after_epoch(self): return True
    def begin_batch(self, xb, yb):
        self.xb,self.yb = xb,yb
        return True
    def after_loss(self, loss):
        self.loss = loss
        return True
    def after_backward(self): return True
    def after_step(self): return True
```

Then the `CallbackHandler` class,

```py
class CallbackHandler():
    def __init__(self,cbs=None):
        # cbs is a list of Callback objects
        self.cbs = cbs if cbs else []

    def begin_fit(self, learn):
        self.learn,self.in_train = learn,True
        learn.stop = False
        res = True
        # Loops through callbacks, `res` means resume
        # In the later Runner implementation this is not needed
        for cb in self.cbs: res = res and cb.begin_fit(learn)
        return res

    def after_fit(self):
        res = not self.in_train
        for cb in self.cbs: res = res and cb.after_fit()
        return res

    def begin_epoch(self, epoch):
        self.learn.model.train()
        self.in_train=True
        res = True
        for cb in self.cbs: res = res and cb.begin_epoch(epoch)
        return res

    def begin_validate(self):
        self.learn.model.eval()
        self.in_train=False
        res = True
        for cb in self.cbs: res = res and cb.begin_validate()
        return res

    def after_epoch(self):
        res = True
        for cb in self.cbs: res = res and cb.after_epoch()
        return res

    def begin_batch(self, xb, yb):
        res = True
        for cb in self.cbs: res = res and cb.begin_batch(xb, yb)
        return res

    def after_loss(self, loss):
        res = self.in_train
        for cb in self.cbs: res = res and cb.after_loss(loss)
        return res

    def after_backward(self):
        res = True
        for cb in self.cbs: res = res and cb.after_backward()
        return res

    def after_step(self):
        res = True
        for cb in self.cbs: res = res and cb.after_step()
        return res

    def do_stop(self):
        try:     return self.learn.stop
        finally: self.learn.stop = False
```

### Callbacks in Action

To demonstrate the ways to use these callbacks, we have

```py
def one_batch(xb, yb, cb):
    if not cb.begin_batch(xb,yb): return
    loss = cb.learn.loss_func(cb.learn.model(xb), yb)
    if not cb.after_loss(loss): return
    loss.backward()
    if cb.after_backward(): cb.learn.opt.step()
    if cb.after_step(): cb.learn.opt.zero_grad()

def all_batches(dl, cb):
    for xb,yb in dl:
        one_batch(xb, yb, cb)
        if cb.do_stop(): return

def fit(epochs, learn, cb):
    if not cb.begin_fit(learn): return
    for epoch in range(epochs):
        if not cb.begin_epoch(epoch): continue
        all_batches(learn.data.train_dl, cb)

        if cb.begin_validate():
            with torch.no_grad(): all_batches(learn.data.valid_dl, cb)
        if cb.do_stop() or not cb.after_epoch(): break
    cb.after_fit()


class TestCallback(Callback):
    def begin_fit(self,learn):
        super().begin_fit(learn)
        self.n_iters = 0
        return True

    def after_step(self):
        self.n_iters += 1
        print(self.n_iters)
        if self.n_iters>=10: self.learn.stop = True
        return True

fit(1, learn, cb=CallbackHandler([TestCallback()]))
"""
1
2
3
4
5
6
7
8
9
10
"""
```

Note: pytorch hooks are a kind a callbacks that can be more granular than these ones, they can be inserted in model forward and backward passes, so we can do something between layers.

### `Runner`: further cleaning it up

We can further refactor this since there are a lot of duplications. Refer to the notebook `04_callbacks` [here](https://github.com/fastai/course-v3/blob/master/nbs/dl2/04_callbacks.ipynb) and check the `Runner` section. It contains some nice Python power user tricks such as enabling something like `self('begin_fit')` by

```py
class Runner():
    ...
    def __call__(self, cb_name):
        for cb in sorted(self.cbs, key=lambda x: x._order):
            f = getattr(cb, cb_name, None)
            if f and f(): return True
        return False
```

This part of the lecture video is worth revisiting for upleveling Python coding skills.

## Annealing

Notebook: `05_anneal`.

Note: Jeremy uses `%debug` in cells with pdb to debug. Check shapes, check the things an object contains, etc.

We define two new callbacks: the `Recorder` to save track of the loss and our scheduled learning rate, and a `ParamScheduler` that can schedule any hyperparameter as long as it's registered in the state_dict of the optimizer.

It's good to use **parameter scheduling** for everything.

```py
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses = [],[]

    def after_batch(self):
        if not self.in_train: return
        self.lrs.append(self.opt.param_groups[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self): plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)

class ParamScheduler(Callback):
    _order=1
    def __init__(self, pname, sched_func):
        self.pname,self.sched_func = pname,sched_func

    def set_param(self):
        # it's called param_groups in pytorch, and layer_groups in fastai
        for pg in self.opt.param_groups:
            pg[self.pname] = self.sched_func(self.n_epochs/self.epochs)

    def begin_batch(self):
        if self.in_train: self.set_param()
```

Trick: We use `partial()` from `functools` and decorators.

### Python partial function

A partial function allows us to call a second function with fixed values in certain arguments. It avoids replicating code.

```py
def power(base, exponent):
    return base**exponent

def squared(base):
    return base ** 2

# The above is bad
# Instead do this
from functools import partial
squared = partial(power, exponent=2)
```

### Python decorators

A decorator is a function that returns another function.

```py
def divide(a, b):
    return a/b

# vs.

def smart_divide(func):
   def inner(a,b):
      print("I am going to divide",a,"and",b)
      if b == 0:
         print("Whoops! cannot divide")
         return

      return func(a,b)
   return inner

@smart_divide
def divide(a,b):
    return a/b

# this is equivalent to
smart_divide(divide)

# Or make it work for any number of arguments with *args, **kargs
def works_for_all(func):
    def inner(*args, **kwargs):
        print("I can decorate any function")
        return func(*args, **kwargs)
    return inner
```

### Annealer decorator

```py
def annealer(f):
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def sched_lin(start, end, pos): return start + pos*(end-start)

f = sched_lin(1, 2)
f(0.3)
# 1.3
```

Jupyter has an advantage over an IDE that when you hit **shift-tab** to check what `sched_lin` takes in, it shows `start, end` because it runs a Python process and knows it's decorated.

Now, using this approach we can define different schedulers

```py
# sched_cos is the default for fastai
@annealer
def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
@annealer
def sched_no(start, end, pos):  return start
@annealer
def sched_exp(start, end, pos): return start * (end/start) ** pos

def cos_1cycle_anneal(start, high, end):
    return [sched_cos(start, high), sched_cos(high, end)]
```

Trick: pytorch tensors can't be plotted directly because they don't have `ndim`, but we can add it ourselves with the line below!

```py
#This monkey-patch is there to be able to plot tensors
torch.Tensor.ndim = property(lambda x: len(x.shape))
```

In the next lesson, we will look at pytorch hooks and other advanced features.
