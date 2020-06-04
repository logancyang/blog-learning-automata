---
toc: true
layout: post
description: fast.ai note series
categories: [note, fastai]
title: "FastAI Lesson 11: Data Block API, and generic optimizer"
comments: true
---

## Layer-wise Sequential Unit Variance: a smart and simple init

Paper: [All You Need is a Good Init](https://arxiv.org/pdf/1511.06422.pdf)

Notebook: `07a_lsuv`

Trick: in deep learning, `Module`s are like a tree, so recursion for finding modules is needed. To concatenate the list of modules in the *recursion*, we can use `sum(list, [])`, beginning with an empty list.

```py
def find_modules(m, cond):
    if cond(m): return [m]
    return sum([find_modules(o,cond) for o in m.children()], [])

mods = find_modules(learn.model, lambda o: isinstance(o,ConvLayer))
```

The code for LSUV

```py
def append_stat(hook, mod, inp, outp):
    d = outp.data
    hook.mean,hook.std = d.mean().item(),d.std().item()

mdl = learn.model.cuda()

with Hooks(mods, append_stat) as hooks:
    mdl(xb)
    for hook in hooks: print(hook.mean,hook.std)

def lsuv_module(m, xb):
    h = Hook(m, append_stat)

    while mdl(xb) is not None and abs(h.mean)  > 1e-3: m.bias -= h.mean
    while mdl(xb) is not None and abs(h.std-1) > 1e-3: m.weight.data /= h.std

    h.remove()
    return h.mean,h.std

for m in mods: print(lsuv_module(m, xb))

%time run.fit(2, learn)
```

While the mean is not near zero: keep subtracting the bias with it.

While the std is not 1, keep dividing the weight by it.

This is the fastai way of initialization, just a loop, no math!

*Note: LSUV is run once at the beginning before training. If you have a small batch size, run it for 5 batches and take the mean.*

## Data Block API

MNIST is a toy problem and it is too easy for serious research. CIFAR-10's problem is that it's 32x32, and it has very different characteristics from large images. Once the images are smaller than 96x96, things are quite different. Yet ImageNet is too large to experiment on.

So Jeremy created new datasets: [Imagenette and Imagewoof](https://github.com/fastai/imagenette). Imagenette (with French pronunciation) is designed to be easier. It has 10 very different classes. Imagewoof has only dog breeds.

*Note: a big part of making deep learning useful in any domain is to make small workable datasets.*

The Data Block API enables us to load huge datasets bit by bit because we can't fit the whole dataset in our RAM.

Trick: add a custom function to any standard library. Just take the class and define a new method (this is the advantage of using a dynamic language)

```py
import PIL,os,mimetypes
Path.ls = lambda x: list(x.iterdir())
path.ls()
```

Joke: If a person says they are a deep learning practioner they must know tenches. Tench is the 1st class in ImageNet.

When we load an image it's size is `(160, 239, 3)` (h, w, ch). The pixels are `uint8`.

Notice it's not square, and the images in this dataset have different sizes. We need to `resize` them in order to put them in the same batch.

*Tip: Python's `os.scandir(path)` is a super fast way to check the directory for content, it's written in C. `os.walk(path)` is similar but it's able to recursively walk the directory. It is faster than `glob` and is lower-level. `glob` has more functionality and should be using `scandir` under the hood.*

For 13394 files, it took ~70ms, extremely fast. The original ImageNet is 100x bigger, it will take just a few seconds.

*Note: Jeremy spent a lot of time of these notebooks in part II, they are his research journal. Much of the code is already in fastai v1.*

### Prepare for modeling

What we need to do:

- Get files
- Split validation set
  - random%, folder name, csv, ...
- Label:
  - folder name, file name/re, csv, ...
- Transform per image (optional)
- Transform to tensor
- DataLoader
- Transform per batch (optional)
- DataBunch
- Add test set (optional)

```py
def compose(x, funcs, *args, order_key='_order', **kwargs):
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x

class ItemList(ListContainer):
    def __init__(self, items, path='.', tfms=None):
        super().__init__(items)
        self.path,self.tfms = Path(path),tfms

    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'

    def new(self, items, cls=None):
        if cls is None: cls=self.__class__
        """cls , or self.__class__, becomes the constructor"""
        return cls(items, self.path, tfms=self.tfms)

    def  get(self, i): return i
    def _get(self, i): return compose(self.get(i), self.tfms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res,list): return [self._get(o) for o in res]
        return self._get(res)

class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = image_extensions
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, fn): return PIL.Image.open(fn)
```

Tip: `compose` is a very useful concept in functional programming, it has a list of functions, calls the current function, get the result, and plug the result into the next function.

### Labeling

We need some kind of `Processor` to do things on the training set and be able to do the same to the validation/test sets later, things such as

- processing texts to tokenize and numericalize them
- filling in missing values with median computed from training set for tabular data, storing the statistics in the `Processor`
- converting label strings to numbers in a consistent and reproducible way

In the image classification case, we create a list of possible labels in the training set, and then convert our labels to numbers based on this *vocab*.

Note: when a trained model is no better than random, the most common issue is that the validation set and the training set have different processing/mapping. Validation set should use the same processing with the same *vocab* and statistics as the training set!

Tip: whatever framework you use, have something like the `fastai` Labeler and Processor classes to help you remember the right things to do.

---
Question: How to make the model handle unseen categories at inference time?

Answer: Great question especially for tasks with unlimited classes. In that case, find some rare classes in your data and label them as "other", train with some examples in the "other" category. That way, the model should be able to hanlde unseen categories better.

---

Note: learn Python `@classmethod`, what they are, when and why to use them. Refer to the post [here]().

Then we resize and turn the images into tensors.

```py
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, splitter)
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
```

## Modeling

### Data Bunch

```py
class DataBunch():
    def __init__(self, train_dl, valid_dl, c_in=None, c_out=None):
        self.train_dl,self.valid_dl,self.c_in,self.c_out = \
            train_dl,valid_dl,c_in,c_out

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset

def databunchify(sd, bs, c_in=None, c_out=None, **kwargs):
    dls = get_dls(sd.train, sd.valid, bs, **kwargs)
    return DataBunch(*dls, c_in=c_in, c_out=c_out)

SplitData.to_databunch = databunchify

"""
Summarize all the steps from the start
"""
path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
tfms = [
    make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor
]

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(
    il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=4)
```

### Model

In CNN, 3x3 kernels are the best bang for your buck. Papers:

- [Visualizing and understanding convolution networks](https://arxiv.org/abs/1311.2901)
- [Bag of tricks for image classification with convolutional neural networks](https://arxiv.org/abs/1812.01187)

```py
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback]

def normalize_chan(x, mean, std):
    return (x-mean[...,None,None]) / std[...,None,None]

_m = tensor([0.47, 0.48, 0.45])
_s = tensor([0.29, 0.28, 0.30])
norm_imagenette = partial(normalize_chan, mean=_m.cuda(), std=_s.cuda())

cbfs.append(partial(BatchTransformXCallback, norm_imagenette))
nfs = [64,64,128,256]

def prev_pow_2(x): return 2**math.floor(math.log2(x))

def get_cnn_layers(data, nfs, layer, **kwargs):
    def f(ni, nf, stride=2): return layer(ni, nf, 3, stride=stride, **kwargs)
    l1 = data.c_in
    l2 = prev_pow_2(l1*3*3)
    layers =  [f(l1  , l2  , stride=1),
               f(l2  , l2*2, stride=2),
               f(l2*2, l2*4, stride=2)]
    nfs = [l2*4] + nfs
    layers += [f(nfs[i], nfs[i+1]) for i in range(len(nfs)-1)]
    layers += [nn.AdaptiveAvgPool2d(1), Lambda(flatten),
               nn.Linear(nfs[-1], data.c_out)]
    return layers

def get_cnn_model(data, nfs, layer, **kwargs):
    return nn.Sequential(*get_cnn_layers(data, nfs, layer, **kwargs))

def get_learn_run(nfs, data, lr, layer, cbs=None, opt_func=None, **kwargs):
    model = get_cnn_model(data, nfs, layer, **kwargs)
    init_cnn(model)
    return get_runner(model, data, lr=lr, cbs=cbs, opt_func=opt_func)

sched = combine_scheds([0.3,0.7], cos_1cycle_anneal(0.1,0.3,0.05))
learn,run = get_learn_run(nfs, data, 0.2, conv_layer, cbs=cbfs+[
    partial(ParamScheduler, 'lr', sched)
])
```

Create `model_summary()`

```py
def model_summary(run, learn, data, find_all=False):
    xb,yb = get_batch(data.valid_dl, run)
    device = next(learn.model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks: learn.model(xb)
```

Training the model,

```py
%time run.fit(5, learn)
"""
train: [1.7975745138242594, tensor(0.3771, device='cuda:0')]
valid: [1.950084228515625, tensor(0.3640, device='cuda:0')]
train: [1.331341733558244, tensor(0.5549, device='cuda:0')]
valid: [1.182614013671875, tensor(0.6160, device='cuda:0')]
train: [1.0004353405653792, tensor(0.6729, device='cuda:0')]
valid: [0.9452028198242187, tensor(0.6740, device='cuda:0')]
train: [0.744675257750698, tensor(0.7583, device='cuda:0')]
valid: [0.8292762451171874, tensor(0.7360, device='cuda:0')]
train: [0.5341721137253761, tensor(0.8359, device='cuda:0')]
valid: [0.798895751953125, tensor(0.7360, device='cuda:0')]
CPU times: user 25.6 s, sys: 10.7 s, total: 36.4 s
Wall time: 1min 7s
"""
```

This is the most basic CNN, and its performance is not bad!

## Optimizers

Notebook: `09_optimizers`

Jeremy: we don't need to re-implement the optimizer every time a new one comes out. **There is only ONE generic optimizer, and we can change it to get every optimizer.**

In pytorch, the optimizer is just a dictionary.

```py
class Optimizer():
    def __init__(self, params, steppers, **defaults):
        """params is a list of lists of parameter tensors"""
        # might be a generator
        self.param_groups = list(params)
        # ensure params is a list of lists
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]
        """
        One dict of hyperparameters for each parameter group.
        This line below copies the defaults dict rather than
        pointing to the same reference.
        Hyperparameters include lr, eps, etc.
        """
        self.hypers = [{**defaults} for p in self.param_groups]
        self.steppers = listify(steppers)

    def grad_params(self):
        return [
            (p,hyper) for pg,hyper in zip(
                self.param_groups,self.hypers)
            for p in pg if p.grad is not None
        ]

    def zero_grad(self):
        for p,hyper in self.grad_params():
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        for p,hyper in self.grad_params():
            compose(p, self.steppers, **hyper)

"""Stepper"""
def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)
    return p

opt_func = partial(Optimizer, steppers=[sgd_step])
```

We can also add weight decay. We can do it in one of two ways:

- Add L2 regularization to the loss, or
- Add weight decay to the gradient `weight.grad += wd * weight`. For a vanilla SGD the update is

```
weight = weight - lr*(weight.grad + wd*weight)
```

These two ways are only equivalent for vanilla SGD. For RMSprop and Adam, the second way is better. It is mentioned in the paper [DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf). `fastai` made the second way the default.

We can implement `stepper` for weight decay.

```py
def weight_decay(p, lr, wd, **kwargs):
    p.data.mul_(1 - lr*wd)
    return p

weight_decay._defaults = dict(wd=0.)

# Or
def l2_reg(p, lr, wd, **kwargs):
    # Tip: Pytorch add_ can take two parameters a, b,
    # it does a mult of a and b first and then add
    p.grad.data.add_(wd, p.data)
    return p

l2_reg._defaults = dict(wd=0.)
```

*Tip: Pytorch `add_` can take two parameters a, b, it does a mult of a and b first and then add the result to the tensor.*

### Add momentum

Momentum needs the previous state of all parameters to work. We store it in a dict `state`.

```py
class StatefulOptimizer(Optimizer):
    def __init__(self, params, steppers, stats=None, **defaults):
        self.stats = listify(stats)
        maybe_update(self.stats, defaults, get_defaults)
        super().__init__(params, steppers, **defaults)
        self.state = {}

    def step(self):
        for p,hyper in self.grad_params():
            if p not in self.state:
                # Create a state for p and call all the
                # statistics to initialize it.
                self.state[p] = {}
                maybe_update(
                    self.stats,
                    self.state[p],
                    lambda o: o.init_state(p)
                )
            state = self.state[p]
            for stat in self.stats:
                state = stat.update(p, state, **hyper)
            compose(p, self.steppers, **state, **hyper)
            self.state[p] = state

class AverageGrad(Stat):
    _defaults = dict(mom=0.9)

    def init_state(self, p):
        return {'grad_avg': torch.zeros_like(p.grad.data)}
    def update(self, p, state, mom, **kwargs):
        state['grad_avg'].mul_(mom).add_(p.grad.data)
        return state

def momentum_step(p, lr, grad_avg, **kwargs):
    p.data.add_(-lr, grad_avg)
    return p

sgd_mom_opt = partial(
    StatefulOptimizer, steppers=[momentum_step,weight_decay],
    stats=AverageGrad(), wd=0.01
)

learn,run = get_learn_run(
    nfs, data, 0.3, conv_layer, cbs=cbfs, opt_func=sgd_mom_opt
)
```

**!!NOTE!! Batch norm mults makes L2 regularization not work the way we expected!**

Everybody has been doing it wrong! This paper [L2 regularization vs batch and weight normalization](https://arxiv.org/abs/1706.05350) pointed it out. But this paper also isn't completely correct. L2 regularization DOES DO SOMETHING. There are some more recent papers that tried to explain what L2 regularization does along with batch norm, but the current state is that *no one understands it completely*.

Jeremy: The theory people who did this kind of research don't know how to train models. The practioners forget about theories. If you can combine the two, you can find interesting results!

- Jane street blog [post](https://blog.janestreet.com/l2-regularization-and-batch-norm/)
- Paper: [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281)

Momentum is also interesting. We use Exponentially Weighted Moving Average (ewma) or `lerp` in pytorch.

Next, the notebook describes the implementation of Adam and LAMB ([paper](https://arxiv.org/abs/1904.00962)) with the generic optimizer structure. Refer to the [notebook](https://github.com/fastai/course-v3/blob/master/nbs/dl2/09_optimizers.ipynb) for more details.

Jeremy implemented the LAMB approach a week before the paper came out. The paper is highly recommended.

## Refactor: remove Runner, just have Learner

Notebook: `09b_learner`

<https://github.com/fastai/course-v3/blob/master/nbs/dl2/09b_learner.ipynb>

Jeremy realized the `Runner` class just stores 3 things and it's much better to just put it in `Learner`. This makes the code much easier to use.

## Add a progress bar

Notebook: `09c_add_progress_bar`

<https://github.com/fastai/course-v3/blob/master/nbs/dl2/09c_add_progress_bar.ipynb>

The components used are

```py
from fastprogress import master_bar, progress_bar
```

And we create the callback

```py
class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats,self.valid_stats = \
            AvgStats(metrics,True),AvgStats(metrics,False)

    def begin_fit(self):
        met_names = ['loss'] + [
            m.__name__ for m in self.train_stats.metrics]
        names = ['epoch'] + [f'train_{n}' for n in met_names] + \
            [f'valid_{n}' for n in met_names] + ['time']
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else \
            self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f'{v:.6f}' for v in o.avg_stats]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)
```

Then we add the progress bars... with a Callback of course! `master_bar` handles the count over the epochs while its child `progress_bar` is looping over all the batches. We just create one at the beginning or each epoch/validation phase, and update it at the end of each batch. By changing the logger of the `Learner` to the `write` function of the master bar, everything is automatically written there.

Note: this requires fastprogress v0.1.21 or later.

```py
# export
class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self): self.mbar.on_iter_end()
    def after_batch(self): self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()

    def set_pb(self):
        self.pb = progress_bar(self.dl, parent=self.mbar)
        self.mbar.update(self.epoch)
```

Then use it like this

```py
cbfs = [partial(AvgStatsCallback,accuracy),
        CudaCallback,
        ProgressCallback,
        partial(BatchTransformXCallback, norm_imagenette)]

learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs)
learn.fit(2)
```

## Data augmentation

Notebook: `10_augmentation`

To further improve our Imagenette model, we need data augmentation.

The key takeaway is that there is no "best transform" for data augmentation. Try things out and take a close look at the results.

Some transform examples:

```py
img.resize((128,128), resample=PIL.Image.ANTIALIAS)

img.resize((128,128), resample=PIL.Image.BILINEAR)

img.resize((128,128), resample=PIL.Image.NEAREST)

img.resize((256,256), resample=PIL.Image.BICUBIC)\
   .resize((128,128), resample=PIL.Image.NEAREST)
```

*Tip: doing transforms on bytes (i.e. uint8) is much faster than on floats. Converting bytes to floats is much slower than something as complex as a warp!*

Some useful transforms that are particularly useful:

- zooming in. It works great for image, for text and audio.
- perspective warping for image. It needs to solve a system of linear equations. Pytorch has this solver!

One comment from Jeremy is that for zooming in, if the object of interest - the tench - is cropped out by the zoom-in effect, it's OK! It's still the ImageNet winning strategy. Ultimately it creates noisy labels where some labels are just wrong. All the research showed that noisy labels are okay - because the model learns to link other things in that image with the label.

For music data augmentation, for example, you can do pitch shifting, volume changes, cutting, etc. Ultimately it depends on the domain and what you need.

In the next lesson, we will introduce *MixUp*, a data augmentation technique that dramatically improves results no matter the domain, and can be run on GPU. It will make some of the techniques here irrelevant.

## Papers to read

- L2 Regularization versus Batch and Weight Normalization
- Norm matters: efficient and accurate normalization schemes in deep networks
- Three Mechanisms of Weight Decay Regularization
- Nesterov's Accelerated Gradient and Momentum as approximations to Regularised Update Descent
- Adam: A Method for Stochastic Optimization
- Reducing BERT Pre-Training Time from 3 Days to 76 Minutes
- Blog post: <https://blog.janestreet.com/l2-regularization-and-batch-norm/>
