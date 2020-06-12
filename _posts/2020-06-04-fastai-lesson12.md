---
toc: true
layout: post
description: fast.ai note series
categories: [note, fastai]
title: "FastAI Lesson 12: Advanced training techniques; ULMFiT from scratch"
comments: true
---

We implement some really important training techniques today, all using callbacks:

- MixUp, a data augmentation technique that dramatically improves results, particularly when you have less data, or can train for a longer time
- Label smoothing, which works particularly well with MixUp, and significantly improves results when you have noisy labels
- Mixed precision training, which trains models around 3x faster in many situations.

We also implement xresnet, which is a tweaked version of the classic resnet architecture that provides substantial improvements. And, even more important, the development of it provides great insights into what makes an architecture work well.

Finally, we show how to implement ULMFiT from scratch, including building an LSTM RNN, and looking at the various steps necessary to process natural language data to allow it to be passed to a neural network.

## Jeremy's starting comments

We haven't done any NLP yet, but NLP and CV share the same code for basic building blocks which we have written.

Comment: for code formatting, Jeremy doesn't like using rules and formatters because he can do his own custom formatting for better readability like this:

```py
def one_batch(self, i, xb, yb):
    try:
        self.iter = i
        self.xb,self.yb = xb,yb;                        self('begin_batch')
        self.pred = self.model(self.xb);                self('after_pred')
        self.loss = self.loss_func(self.pred, self.yb); self('after_loss')
        if not self.in_train: return
        self.loss.backward();                           self('after_backward')
        self.opt.step();                                self('after_step')
        self.opt.zero_grad()
    except CancelBatchException:                        self('after_cancel_batch')
    finally:                                            self('after_batch')
```

The goal of formatting is to make the code more readable, and debugging ML code is very difficult. This kind of formatting can help.

## MixUp and label smoothing

In the last lesson we can run data augmentation on GPUs.

Now, we can use something called MixUp that makes other data augmentation irrelevant. It's in the paper:

[mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

Highly recommended read.

Tip: Using Greek letters in Python code makes it easy to check the code against the paper. Python supports unicode.

```py
class MixUp(Callback):
    _order = 90 #Runs after normalization and cuda
    def __init__(self, α:float=0.4): self.distrib = Beta(tensor([α]), tensor([α]))

    def begin_fit(self): self.old_loss_func,self.run.loss_func = self.run.loss_func,self.loss_func

    def begin_batch(self):
        if not self.in_train: return #Only mixup things during training
        λ = self.distrib.sample((self.yb.size(0),)).squeeze().to(self.xb.device)
        λ = torch.stack([λ, 1-λ], 1)
        self.λ = unsqueeze(λ.max(1)[0], (1,2,3))
        shuffle = torch.randperm(self.yb.size(0)).to(self.xb.device)
        xb1,self.yb1 = self.xb[shuffle],self.yb[shuffle]
        # Doing a linear combination on the images
        self.run.xb = lin_comb(self.xb, xb1, self.λ)

    def after_fit(self): self.run.loss_func = self.old_loss_func

    def loss_func(self, pred, yb):
        if not self.in_train: return self.old_loss_func(pred, yb)
        with NoneReduce(self.old_loss_func) as loss_func:
            loss1 = loss_func(pred, yb)
            loss2 = loss_func(pred, self.yb1)
        # Doing a linear combination on the losses
        loss = lin_comb(loss1, loss2, self.λ)
        return reduce_loss(loss, getattr(self.old_loss_func, 'reduction', 'mean'))
```

MixUp applies linear combination in the form `v1 * α + v2 * (1 - α)`
(previous we used it for Exponentially Weighted Moving Average) on images and losses.

We can use MixUp not just to the input layer. We can use it in the 1st layer, for example. It's not well researched yet.

## Label smoothing: a cure for noisy labels

Softmax wants to produce one number that is very close to 1. With MixUp and label noise in the dataset, we don't want 100% certainty in the output. Use **label smoothing, a regularization technique**.

*Tip: Don't wait until there is a perfectly labeled dataset to start modelling. Use **label smoothing**. It works well even with noisy labels!*

Label smoothing is designed to make the model a little bit less certain of it's decision by changing a little bit its target: instead of wanting to predict 1 for the correct class and 0 for all the others, we ask it to predict `1-ε` for the correct class and `ε` for all the others, with `ε` a (small) positive number and N the number of classes. This can be written as:

$$loss = (1-ε) ce(i) + ε \sum ce(j) / N$$

where `ce(x)` is cross-entropy of `x` (i.e. $-\log(p_{x})$), and `i` is the correct class. This can be coded in a
loss function:

```py
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.ε)
```

And we use it by

```py
learn = get_learner(nfs, data, 0.4, conv_layer, cb_funcs=cbfs, loss_func=LabelSmoothingCrossEntropy())
learn.fit(1)
```

## Train in half precision floating point

Notebook: `10c_fp16`

We use NVidia's `apex` library to use half precision floating point so we can make training much faster. To avoid inaccurate computation, we only make the forward and backward passes use half precision, and use full precision for everywhere else.

Refer to the notebook and the lecture for more details.

## xresnet: train Imagenette

Notebook: `11_train_imagenette`

This is like ResNet but there are a few tweaks.

- The 1st tweak is called ResNet-C. The idea is that instead of using 7x7 kernel we use 3 times 3x3 kernel.
- 2nd tweak is that we initialize the batch norm to sometimes have weights of 0 and sometimes weights of 1. The idea behind this is that then sometimes ResBlock can be ignored. This way we can train very deep models with high learning rates because if the model doesn’t need the layer it can skip it by keeping the batch norm weights to zero.
- 3rd tweak is to move stride two one convolution up.

The paper Jeremy is talking about: [bag of tricks](https://arxiv.org/abs/1812.01187)

Big companies try to brag with how big batches they can train once. For us, normal people, increasing the learning rate is something we want. That way we can speed training and generalize better.

Jeremy showed how using these techniques he made 3rd best ImageNet model. The two models above this are much bigger and require a lot of computation power.

Comment: for both research and production, code refactoring is very important!

```py
def cnn_learner(arch, data, loss_func, opt_func, c_in=None, c_out=None,
                lr=1e-2, cuda=True, norm=None, progress=True, mixup=0, xtra_cb=None, **kwargs):
    cbfs = [partial(AvgStatsCallback,accuracy)]+listify(xtra_cb)
    if progress: cbfs.append(ProgressCallback)
    if cuda:     cbfs.append(CudaCallback)
    if norm:     cbfs.append(partial(BatchTransformXCallback, norm))
    if mixup:    cbfs.append(partial(MixUp, mixup))
    arch_args = {}
    if not c_in : c_in  = data.c_in
    if not c_out: c_out = data.c_out
    if c_in:  arch_args['c_in' ]=c_in
    if c_out: arch_args['c_out']=c_out
    return Learner(arch(**arch_args), data, loss_func, opt_func=opt_func, lr=lr, cb_funcs=cbfs, **kwargs)

lr = 1e-2
pct_start = 0.5

def create_phases(phases):
    phases = listify(phases)
    return phases + [1-sum(phases)]

phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95, 0.85, 0.95))
cbsched = [
    ParamScheduler('lr', sched_lr),
    ParamScheduler('mom', sched_mom)]

learn = cnn_learner(xresnet34, data, loss_func, opt_func, norm=norm_imagenette)
learn.fit(5, cbsched)
```

## Transfer learning from scratch

Now we illustrate how to do transfer learning from scratch using an ImageWoof model on the Pets task. The ImageWoof model is small and only has 10 classes, and the Pets data has 37 classes and it has cats. So it is an unusual case for transfer learning, but a very interesting experiment to see whether it works.

1. Train a xresnet model for 40 epochs from scratch on ImageWoof, save the model. (valid accuracy ~80%)
2. Inspect how well xresnet model does on another task, Pets, by training for 1 epoch from scratch. (valid accuracy ~30%)
3. Custom the head
   1. Load the saved ImageWoof model for Pets, set the `learner` to have `c_out=10` because ImageWoof has 10 classes; the learner now has ImageWoof model pointing to the Pets databunch.
   2. Remove the output linear layer which has 10 activations, and replace it with a linear layer with 37 activations according to the Pets data.
      1. This is tricky because we don't know the input dimensions of this new linear layer. We need to find out by looking at the previous layer's output shape. Specifically, we pass a batch of data through the cut down model (without the head), and look the output shape.
   3. The transferred model: `nn.Sequential(m_cut, AdaptiveConcatPool2d(), Flatten(),nn.Linear(ni*2, data.c_out))`. The `AdaptiveConcatPool2d()` is something fastai has been using for a long time, somebody recently wrote a paper about it. It gives a nice boost. With this, the output linear layer needs 2x number of inputs because it has 2 kinds of pooling.
4. DON'T DO THIS. Naive finetuning: train without freezing the body.
5. DO THIS. Correct finetuning:
   1. train the model with new head **with body frozen**.
   2. Unfreeze the body and train some more. Something weird happens.
      1. The frozen body of the ImageWoof model has frozen batch norm layers where they have means and stds that are not compatible to the Pets data.
      2. Solution: **freeze non-batchnorm layers only!**

### Saving the model

After we train a model on the initial ImageWoof dataset, we need to save it. To save the model means we need to save the weights, and the weights are in `model.state_dict()` which is an `OrderedDict` that has keys as `<layer_name>.weight` and `<layer_name>.bias`.

 Use `torch.save(<state_dict>, <path>)` to save the model. We can also use `Pickle`. **`torch.save` also uses `Pickle` behind the scenes, it just adds some metadata about the model version and type info.**

```py
st = st = learn.model.state_dict()
mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)
torch.save(st, mdl_path/'iw5')
```

Tip: If you have trouble loading something, just try `torch.load(...)` into a dict and take a look at the keys and values and check what's wrong.

```py
# This is a Pets learner, with `data` as pets data
# Note that c_out is changed to 10 to be able to load ImageWoof model
learn = cnn_learner(
    xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
st = torch.load(mdl_path/'iw5')
m = learn.model
m.load_state_dict(st)
```

Tip: In Jupyter Notebook, select multiple cells, `c` to copy, `v` to paste, `shift+m` to merge into one cell, add a function header and we have a function from previous cells!

### Custom the head

Terminology: **Body** means the part that is transferred, in other words, the initial frozen part of the model. **Head** means the part that we train, the output layer(s).

The code below takes the ImageWoof model and adapts it for the task: Pets.

```py
def adapt_model(learn, data):
    # Find everything before the AdaptiveAvgPool2d layer
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    # Get the num of channels `ni` before average pooling
    m_cut = learn.model[:cut]
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    # New model. We added `AdaptiveConcatPool2d(), Flatten()`
    m_new = nn.Sequential(
        m_cut, AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new
```

To use these pretrained weights we need to remove the last linear layer and replace it another layer that have the right number of outputs.

```py
"""
Since learn.model is nn.Sequential(), [0] is the body.
This line means we freeze the parameters in the body.
"""
for p in learn.model[0].parameters(): p.requires_grad_(False)
```

### Don't freeze batch norm layers!

The important thing to notice when fine-tuning is that *batch norm will mess the accuracy because it is frozen with the statistics for a different task*. The solution to this is to only freeze the layers that don’t contain batch norm.

```py
def apply_mod(m, f):
    """
    Nice little function that recursively applies f to all children of m
    """
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)
```

Trick: pytorch has the same recursive functionality of `apply_mod(model, func)`, it is `model.apply(func)`.

---
Notice: `require_grad=False` means **freezing**, not updating the tensor. Pay attention to it in the `micrograd` framework for better understanding of *autograd* and its recursion.
---

---

### Discriminative learning rate and parameter groups

**Discriminative learning rate** is an approach to freeze some layers without setting `require_grad=False`, but set the learning rate to 0 for these layers.

Look at the code below. It groups parameters into 2 groups `g1` and `g2`, `g2` for batch norm layers, and anything else with weights to `g1`. And it does it recursively.

```py
def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters()
        elif hasattr(l, 'weight'): g1 += l.parameters()
        for ll in l.children(): _bn_splitter(ll, g1, g2)

    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)

    g2 += m[1:].parameters()
    return g1,g2
```

This is one of those things that if you got wrong, the model won't train correctly but you won't get an error. That's very hard to debug. So we need a debug callback to look into the model.

*Assume the code you write is wrong, have a good test strategy!*

Jeremy introduced a `DebugCallback` that overwrites the `__call__()` method, and it works for any callbacks with `cb_name` passed in.

```py
class DebugCallback(Callback):
    _order = 999
    def __init__(self, cb_name, f=None): self.cb_name,self.f = cb_name,f
    def __call__(self, cb_name):
        if cb_name==self.cb_name:
            if self.f: self.f(self.run)
            else:      set_trace()

# Usage
def _print_det(o):
    print (len(o.opt.param_groups), o.opt.hypers)
    raise CancelTrainException()

learn.fit(1, disc_lr_sched + [DebugCallback(cb_types.after_batch, _print_det)])
```

Refer to the notebook and the lecture at around 1:07:00 for more details.

### Jeremy on his stance against cross validation

Cross validation is a good idea when the data is very small. Once the data has over 1k rows, it is not needed. In general, if the valid accuracy varies too much from run to run, consider cross validation. Otherwise, a good validation set is enough.

### Best tips for debugging deep learning

- **Don't make mistakes in the first place: make the code as simple as possible.**
- A horror story from Jeremy: he forgot to write `.opt` somewhere and it took countless hours and $5k of AWS credit to find that bug.
- Testing for DL is different from standard software engineering, it needs to work for randomness. Working for one random seed doesn't mean it works for another. You need non-reproducible tests, you need to be warned if something looks off statistically.
- Once you realize there's a specific bug, you write a test that fails on it **everytime**.
- DL debugging is really hard, again, need to **make sure you don't make a mistake in the first place**.

## ULMFiT is transfer learning applied to AWD-LSTM

tbd

## Papers

- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [Rethinking the Inception Architecture for Computer Vision ](https://arxiv.org/abs/1512.00567) (label smoothing is in part 7)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
