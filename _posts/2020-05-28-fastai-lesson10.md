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

## Softmax

A recap of the softmax function and the multiclass cross entropy loss.

In code, log softmax is

```py
def log_softmax(x):
    return x - x.exp().sum(-1, keepdim=True).log()
```

Here `x` is the activation vector, `log_softmax(x)` is a vector with the same shape as `x`.

In equation it is (i for the ith element of one prediction vector)

$$\hat{y}_i = \text{softmax}(\mathbf{x})_{i} = e^{x_{i}} / \sum_{j} e^{x_{j}}$$

$$\text{logsoftmax}(\mathbf{x})_{i} = x_{i} - \log \sum_{j} e^{x_{j}}$$

And cross entropy loss (NLL) for $\mathbf{x}$, i.e. the activation vector of **one** prediction is:

$$-\log(\hat{y}_i)$$

This is because the ground truth `y` is one-hot encoded. Refer to lesson 9's [note](http://blog.logancyang.com/note/fastai/2020/05/25/fastai-lesson9.html#create-the-cross-entropy-loss-function) to recall the **selection trick**.

For multiple predictions, recall that the cross entropy loss or NLL is

```py
def nll(softmax_preds, targets):
    """
    Use array indexing to select the corresponding values for
    cross entropy loss.
    """
    log_sm = softmax_preds.log()
    return -log_sm[range(targets.shape[0]), targets].mean()
```

The `mean()` is for averaging over multiple rows of log softmax predictions to get an overall batch prediction loss.

### When to use softmax and when not to

**Softmax likes to pick one thing and make it big**, because it's exponential.

<img src="{{ site.baseurl }}/images/fastai/softmax_excel.png" alt="softmax_excel" align="middle"/>

In the above Excel example, the activations for these categories in image 1 are larger than in image 2, which means **image 1 is more likely to have these objects in it (Me: this teaches us that the activations before the softmax express confidence of having those things in the image)**.

But, the softmax outputs are the same because after the `exp()` and the normalization, each component captures the same percentage.

**Yet they are different**.

### Very important remarks by Jeremy

---
*Be careful when softmax is a BAD IDEA:*

*To use softmax, make sure that the entries in your dataset all have one or more objects of interests, PREFERRABLY ONE OF EACH TYPE. If none of the images have the objects of interest in them, softmax will still give a high probability of seeing them! If a category has more than one object in an image, softmax finds the most likely ONE. This also applies to audio or tabular data.*

*For yes or no (**whether there is** an object of type A or B or C in the image) kind of tasks, **we should use sigmoid instead of softmax**, as shown at the far right in the above Excel example (note that they don't sum to one anymore).*

*Why did we always use softmax in object recognition tasks? Because of ImageNet! The data entries in ImageNet always have ONE of some object of interest in them!!*

*A lot of well-regarded academic papers or applications use `Nothing` as a category alongside others like `Cat`, `Fish`, etc. But Jeremy says **it's terrible idea**! Because there is no feature like "furriness" or "smoothness" or "shininess" that describes "No-Cat", "No-Fish", etc. Of course we can hack it by somehow producing another model that captures the "none-cat-ness" features but that is too hard and unnecessary. **Just use a binary model** for predicting whether there's an object in the scene!*

**Me: Again, this lesson teaches us that the activations before the last classification outout layer is a monotonic function that indicates the confidence of predicting that category.**

*When you see a paper that uses softmax for classifying exist/non-exist tasks, try to use a sigmoid, you may get better result!*

**When is softmax a good idea? Language modeling!** Predicting the next word is the perfect case for using softmax because it's always one word and no more or less than one word.

---

## Build a Learning Rate Finder

Notebook: `05b_early_stopping`

### Using Exceptions as control flow!

It is not easy to use callbacks and a boolean stop value to do early stopping because we need to check many places. Using Exception is a neat trick.

An exception in Python is just a class that inherits from `Exception`. Most of the time you don't need to give it any behavior, just pass, like this,

```py
class CancelTrainException(Exception): pass
```

We have the `Runner` class and the `Callback` class.

```py
class Callback():
    _order=0
    def set_runner(self, run): self.run=run
    def __getattr__(self, k): return getattr(self.run, k)

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        return camel2snake(name or 'callback')

    def __call__(self, cb_name):
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False

class TrainEvalCallback(Callback):
    def begin_fit(self):
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter   += 1

    def begin_epoch(self):
        self.run.n_epochs=self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False

class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass

#########################

class Runner():
    def __init__(self, cbs=None, cb_funcs=None):
        cbs = listify(cbs)
        for cbf in listify(cb_funcs):
            cb = cbf()
            setattr(self, cb.name, cb)
            cbs.append(cb)
        self.stop,self.cbs = False,[TrainEvalCallback()]+cbs

    @property
    def opt(self):       return self.learn.opt
    @property
    def model(self):     return self.learn.model
    @property
    def loss_func(self): return self.learn.loss_func
    @property
    def data(self):      return self.learn.data

    def one_batch(self, xb, yb):
        try:
            self.xb,self.yb = xb,yb
            self('begin_batch')
            self.pred = self.model(self.xb)
            self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb)
            self('after_loss')
            if not self.in_train: return
            self.loss.backward()
            self('after_backward')
            self.opt.step()
            self('after_step')
            self.opt.zero_grad()
        except CancelBatchException: self('after_cancel_batch')
        finally: self('after_batch')

    def all_batches(self, dl):
        self.iters = len(dl)
        try:
            for xb,yb in dl: self.one_batch(xb, yb)
        except CancelEpochException: self('after_cancel_epoch')

    def fit(self, epochs, learn):
        self.epochs,self.learn,self.loss = epochs,learn,tensor(0.)

        try:
            for cb in self.cbs: cb.set_runner(self)
            self('begin_fit')
            for epoch in range(epochs):
                self.epoch = epoch
                if not self('begin_epoch'): self.all_batches(self.data.train_dl)

                with torch.no_grad():
                    if not self('begin_validate'): self.all_batches(self.data.valid_dl)
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        finally:
            self('after_fit')
            self.learn = None

    def __call__(self, cb_name):
        res = False
        for cb in sorted(self.cbs, key=lambda x: x._order): res = cb(cb_name) or res
        return res
```

We see that `CancelBatchException`, `CancelEpochException` and `CancelTrainException` are used as control flow to enable graceful skip or stopping, by placing it with `except` between `try` and `finally` blocks.

We can use `CancelTrainException` to make a learning rate finder,

```py
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
        self.best_loss = 1e9

    def begin_batch(self):
        if not self.in_train: return
        pos = self.n_iter/self.max_iter
        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
        for pg in self.opt.param_groups: pg['lr'] = lr

    def after_step(self):
        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
            raise CancelTrainException()
        if self.loss < self.best_loss: self.best_loss = self.loss
```

In `after_step()` we check if the loss gets much worse, if yes we stop training.

## Recreate CNN (CPU and GPU)

Notebook: `06_cuda_cnn_hooks_init`

```py
# MNIST
x_train,y_train,x_valid,y_valid = get_data()

# Normalize based on training data
def normalize_to(train, valid):
    m,s = train.mean(),train.std()
    return normalize(train, m, s), normalize(valid, m, s)

x_train,x_valid = normalize_to(x_train,x_valid)
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)

nh,bs = 50,512
c = y_train.max().item()+1
loss_func = F.cross_entropy

data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)

class Lambda(nn.Module):
    """This is for putting into pytorch nn.Sequential()"""
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x): return self.func(x)
```

To refactor layers, it's useful to have a `Lambda` layer that can take a basic function and convert it to a layer you can put in `nn.Sequential`.

*Note: if you use a Lambda layer with a lambda function, your model won't pickle so you won't be able to save it with PyTorch. So it's best to give a name to the function you're using inside your Lambda (like flatten here).*

```py
def flatten(x):
    """Flatten after nn.AdaptiveAvgPool2d and before the final nn.Linear"""
    return x.view(x.shape[0], -1)

def mnist_resize(x):
    """
    Resize bs x 784 to batches of 28x28 images. -1 means the batch size
    remains whatever it is before
    """
    return x.view(-1, 1, 28, 28)
```

Create the CNN model,

```py
def get_cnn_model(data):
    return nn.Sequential(
        # This lambda layer is preprocessing original bs x 784 to
        # bs x 1 x 28 x 28
        Lambda(mnist_resize),
        nn.Conv2d( 1, 8, 5, padding=2,stride=2), nn.ReLU(), #14
        nn.Conv2d( 8,16, 3, padding=1,stride=2), nn.ReLU(), # 7
        nn.Conv2d(16,32, 3, padding=1,stride=2), nn.ReLU(), # 4
        nn.Conv2d(32,32, 3, padding=1,stride=2), nn.ReLU(), # 2
        nn.AdaptiveAvgPool2d(1),
        Lambda(flatten),
        nn.Linear(32,data.c)
    )
```

Now run the model on CPU,

```py
model = get_cnn_model(data)
# Callbacks from previous notebook
cbfs = [Recorder, partial(AvgStatsCallback,accuracy)]

opt = optim.SGD(model.parameters(), lr=0.4)
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)

%time run.fit(1, learn)
"""
train: [1.7832209375, tensor(0.3780)]
valid: [0.68908681640625, tensor(0.7742)]
CPU times: user 7.84 s, sys: 5.79 s, total: 13.6 s
Wall time: 5.87 s
"""
```

This is a bit slow, let's run it on GPU!

### Move to GPU: CUDA

A somewhat flexible way:

```py
# 0 means you have 1 GPU
device = torch.device('cuda', 0)

class CudaCallback(Callback):
    """pytorch has .to(device) for model and tensors"""
    def __init__(self,device):
        self.device=device

    def begin_fit(self):
        self.model.to(self.device)

    def begin_batch(self):
        self.run.xb,self.run.yb = \
            self.xb.to(self.device),self.yb.to(self.device)
```

A less flexible but more convenient way if you only have 1 GPU:

```py
# This only needs to be called once, and pytorch defaults to it
torch.cuda.set_device(device)

class CudaCallback(Callback):
    """Now instead of .to(device), just do .cuda()"""
    def begin_fit(self): self.model.cuda()
    def begin_batch(self): self.run.xb,self.run.yb = self.xb.cuda(),self.yb.cuda()

cbfs.append(CudaCallback)

model = get_cnn_model(data)
opt = optim.SGD(model.parameters(), lr=0.4)
learn = Learner(model, opt, loss_func, data)
run = Runner(cb_funcs=cbfs)
%time run.fit(3, learn)
"""
train: [1.8033628125, tensor(0.3678, device='cuda:0')]
valid: [0.502658544921875, tensor(0.8599, device='cuda:0')]
train: [0.3883639453125, tensor(0.8856, device='cuda:0')]
valid: [0.205377734375, tensor(0.9413, device='cuda:0')]
train: [0.17645265625, tensor(0.9477, device='cuda:0')]
valid: [0.15847452392578126, tensor(0.9543, device='cuda:0')]
CPU times: user 4.36 s, sys: 1.07 s, total: 5.43 s
Wall time: 5.41 s
"""
```

This is much faster than CPU! For a much deeper model, it will be even faster.

### Refactoring the model

First we can regroup all the conv/relu in a single function:

```py
def conv2d(ni, nf, ks=3, stride=2):
    return nn.Sequential(
        nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride), nn.ReLU())
```

We can do the mnist resize in a batch transform, that we can do with a Callback.

```py
class BatchTransformXCallback(Callback):
    _order=2
    def __init__(self, tfm): self.tfm = tfm
    def begin_batch(self): self.run.xb = self.tfm(self.xb)

def view_tfm(*size):
    """
    Using closure to create a view or reshape to `size` with any batch size
    """
    def _inner(x): return x.view(*((-1,)+size))
    return _inner

mnist_view = view_tfm(1,28,28)
cbfs.append(partial(BatchTransformXCallback, mnist_view))
```

Get familiar with closure and partial with the above code.

This model can now work on any size input,

```py
nfs = [8,16,32,32]

def get_cnn_layers(data, nfs):
    nfs = [1] + nfs
    return [
        conv2d(nfs[i], nfs[i+1], 5 if i==0 else 3)
        for i in range(len(nfs)-1)
    ] + [nn.AdaptiveAvgPool2d(1), Lambda(flatten), nn.Linear(nfs[-1], data.c)]

def get_cnn_model(data, nfs): return nn.Sequential(*get_cnn_layers(data, nfs))

#export
def get_runner(model, data, lr=0.6, cbs=None, opt_func=None, loss_func = F.cross_entropy):
    if opt_func is None: opt_func = optim.SGD
    opt = opt_func(model.parameters(), lr=lr)
    learn = Learner(model, opt, loss_func, data)
    return learn, Runner(cb_funcs=listify(cbs))

model = get_cnn_model(data, nfs)
learn,run = get_runner(model, data, lr=0.4, cbs=cbfs)

model
"""
Sequential(
  (0): Sequential(
    (0): Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
    (1): ReLU()
  )
  (1): Sequential(
    (0): Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
  )
  (2): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
  )
  (3): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
  )
  (4): AdaptiveAvgPool2d(output_size=1)
  (5): Lambda()
  (6): Linear(in_features=32, out_features=10, bias=True)
)
"""
run.fit(3, learn)
"""
train: [1.90592640625, tensor(0.3403, device='cuda:0')]
valid: [0.743217529296875, tensor(0.7483, device='cuda:0')]
train: [0.4440590625, tensor(0.8594, device='cuda:0')]
valid: [0.203494482421875, tensor(0.9409, device='cuda:0')]
train: [0.1977476953125, tensor(0.9397, device='cuda:0')]
valid: [0.13920831298828126, tensor(0.9606, device='cuda:0')]
"""
```

## Hooks

### Manual insertion

Having our own Sequential, we can store each layer activations' mean and standard deviation.

```py
class SequentialModel(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.act_means = [[] for _ in layers]
        self.act_stds  = [[] for _ in layers]

    def __call__(self, x):
        for i,l in enumerate(self.layers):
            x = l(x)
            self.act_means[i].append(x.data.mean())
            self.act_stds [i].append(x.data.std ())
        return x

    def __iter__(self): return iter(self.layers)


model =  SequentialModel(*get_cnn_layers(data, nfs))
learn,run = get_runner(model, data, lr=0.9, cbs=cbfs)
run.fit(2, learn)
```

When we plot the means and stds for the layer activations over the training process, we see they explode and drop off a cliff several times. That is really concerning. We don't know if the parameters are stuck in zero gradient places and never come back, and only a small number of them are training.

### Pytorch hooks

Pytorch call them "hooks", we have been calling them "callbacks".

**pytorch hooks == callbacks**

A minimal example,

```py
model = get_cnn_model(data, nfs)
learn,run = get_runner(model, data, lr=0.5, cbs=cbfs)
# Global vars. We can use a Hook class to avoid this.
act_means = [[] for _ in model]
act_stds  = [[] for _ in model]

def append_stats(i, mod, inp, outp):
    """
    A hook is attached to a layer, and needs to have a function that
    takes three arguments: module, input, output. Here we store the
    mean and std of the output in the correct position of our list.
    """
    act_means[i].append(outp.data.mean())
    act_stds [i].append(outp.data.std())


for i,m in enumerate(model):
    # Check the pytorch doc for register_forward_hook() for more details
    m.register_forward_hook(partial(append_stats, i))


run.fit(1, learn)
"""
train: [2.2561553125, tensor(0.1835, device='cuda:0')]
valid: [2.00057578125, tensor(0.3186, device='cuda:0')]

(now act_means, act_stds are populated)
"""
```

Check the notebook's section for the `Hook` class and `Hooks` class for better implementation.

*Tip: When registered hooks, don't forget to remove them when not needed, or you will run out of memory.*

Use the hook with the `with` block like this:

```py
for l in model:
    if isinstance(l, nn.Sequential):
        init.kaiming_normal_(l[0].weight)
        l[0].bias.data.zero_()

with Hooks(model, append_stats) as hooks:
    run.fit(2, learn)
    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    for h in hooks:
        ms,ss = h.stats
        ax0.plot(ms[:10])
        ax1.plot(ss[:10])
    plt.legend(range(6));

    fig,(ax0,ax1) = plt.subplots(1,2, figsize=(10,4))
    for h in hooks:
        ms,ss = h.stats
        ax0.plot(ms)
        ax1.plot(ss)
    plt.legend(range(6));
```

**Python tip: What the `with` block does is that, it calls the `__exit__()` method on the object, in this case `hooks`, after the block**.

After using `kaiming_normal_`, we see that the rise and drop problem is fixed. But what we are really interested in is that, did many activations get super small? Were they nicely activated?

For that, we can add some more statistics into the hooks.

It turns out that after adding histograms and percentage of small activations, we see that over 90% of our activations are wasted (dead ReLU). This is really concerning.

### Generalized ReLU

To avoid wasting most our activations, we can generalize the ReLU by

- leaky ReLU
- subtract by a number and move it into the negatives a bit
- cap it with some max value

Note: `kaiming_normal_` and `kaiming_uniform_` perform similarly for this model. Some people think uniform does better because it has less around 0, but not rigorously studied yet.

## Batch Normalization

Notebook: `07_batchnorm`

Here is the code for batch norm:

```py
class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        # NB: pytorch bn mom is opposite of what you'd expect
        self.mom,self.eps = mom,eps
        # mults and adds are like weights and biases, they are the
        # parameters of the model that we need to learn.
        # They are the beta and gamma in the batch norm paper
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds  = nn.Parameter(torch.zeros(nf,1,1))
        # nn.Module.register_buffer(var, tensor) is the same as
        # var = tensor, but it does more. It automatically moves
        # things to GPU, and it saves them in the model for future use
        self.register_buffer('vars',  torch.ones(1,nf,1,1))
        self.register_buffer('means', torch.zeros(1,nf,1,1))

    def update_stats(self, x):
        # mean and var over dim 0, 2, 3, meaning over batch, width,
        # and height of the images. The result is that each channel/filter
        # has one number for the mean and for the variance
        m = x.mean((0,2,3), keepdim=True)
        v = x.var ((0,2,3), keepdim=True)
        # lerp means linear interpolation
        self.means.lerp_(m, self.mom)
        self.vars.lerp_(v, self.mom)
        return m,v

    def forward(self, x):
        if self.training:
            with torch.no_grad(): m,v = self.update_stats(x)
        else: m,v = self.means,self.vars
        x = (x-m) / (v+self.eps).sqrt()
        return x*self.mults + self.adds
```

The `lerp` part is the exponentially weighted moving average. We define a momentum `mom = 0.9`, say we have a sequence `[3, 5, 4, ...]`, the moving average is

```py
mu1 = 3
mu2 = 0.9 * mu1 + 5 * 0.1
mu3 = 0.9 * mu2 + 4 + 0.1
...

"""
This is the way of calculating the moving average:

        mu_n = mom * mu_{n-1} + new_val * (1 - mom)

This is a way of linear interpolation (lerp)

        a * beta + b * (1-beta)

So the moving average is equivalent to

        m.lerp(new_val, mom)
"""
```

Refer to the above for the definition of the moving average and lerp.

Note: pytorch's `lerp`'s momentum is the exact opposite of the momentum we just defined, so it's a momentum of 0.1 for the case above where we have 0.9. Hence, pytorch's batchnorm has momentum opposite to the momentum normally defined in the optimizers. (Refer to the note [here](https://pytorch.org/docs/stable/nn.html#batchnorm1d))

**After applying batch norm, we have gotten rid of the rise and crash in the means and stds during training entirely!**

<img src="{{ site.baseurl }}/images/fastai/batch_norm_training.png" alt="batch_norm_training" align="middle"/>

### Batch norm deficiencies

**Note: We cannot use batch norm for ONLINE LEARNING and SEGMENTATION because of small batch size, the variance is infinity or unstable, and we can't use it for RNNs.**

The [layer norm paper](https://arxiv.org/abs/1607.06450) proposed the solution to this. The entire paper is essentially this:

```py
class LayerNorm(nn.Module):
    __constants__ = ['eps']
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.))
        self.add  = nn.Parameter(tensor(0.))

    def forward(self, x):
        # The only change compared to batchnorm is
        # instead of (0, 2, 3), we have mean and var over dim (1,2,3)
        # and we don't have moving averages. That's it!
        m = x.mean((1,2,3), keepdim=True)
        v = x.var ((1,2,3), keepdim=True)
        x = (x-m) / ((v+self.eps).sqrt())
        return x*self.mult + self.add
```

Layer norm helps, but it's not as useful as batch norm. But for RNNs, layer norm is the only thing to use.

There are other attempts to work around this, such as instance norm (for style transfer) and group norm. Check out the [group norm paper](https://arxiv.org/pdf/1803.08494.pdf) for details.

<img src="{{ site.baseurl }}/images/fastai/all_norms.png" alt="all_norms" align="middle"/>

However, none of them are as good as batch norm. Jeremy says he doesn't know how to fix it for RNNs, but for small batch size, he has some idea: use `eps`!

```py
class BatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        ...
```

It prevents the numbers to blow up.

A better idea: new algorithm for running batch norm! Visit the notebook section 4 and watch the [video](https://course.fast.ai/videos/?lesson=10&t=8068) for more details. The keyword is *debiasing*.

## Ablation study in deep learning research

Jeremy mentioned ablation study briefly. It is good to know

<https://stats.stackexchange.com/questions/380040/what-is-an-ablation-study-and-is-there-a-systematic-way-to-perform-it>

## My Random Thoughts

It is getting really hardcore in part II lessons! The material has great quality and quatity, extremely rare to find even in top universities. Jeremy is really doing great work for DL learners around the world!

The lessons are great practical lessons to learn

- Advanced Python
- Pytorch fundamentals
- Software engineering
- Turning paper into code
- Code-first research methodology

My goal is to be able to **use** the `fastai` library effectively, and **implement** things in its style effectively. Then I can even become a fastai contributor.
