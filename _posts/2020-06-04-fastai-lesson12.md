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

## Jeremy's answer to some in-class questions

### Jeremy on cross validation

Cross validation is a good idea when the data is very small. Once the data has over 1k rows, it is not needed. In general, if the valid accuracy varies too much from run to run, consider cross validation. Otherwise, a good validation set is enough.

### Best tips for debugging deep learning

- **Don't make mistakes in the first place: make the code as simple as possible.**
- A horror story from Jeremy: he forgot to write `.opt` somewhere and it took countless hours and $5k of AWS credit to find that bug.
- Testing for DL is different from standard software engineering, it needs to work for randomness. Working for one random seed doesn't mean it works for another. You need non-reproducible tests, you need to be warned if something looks off statistically.
- Once you realize there's a specific bug, you write a test that fails on it **everytime**.
- DL debugging is really hard, again, need to **make sure you don't make a mistake in the first place**.

### Scientific journal: all scientists should have one

A note that has all the scientific experiment settings and their results. It should be easy to search through if we need some old records.

For example, noble gases and penicillin are discovered by good practice of scientific journaling.

Jeremy tried changing batch norm in different ways in Keras. His journal helped him keep track of all the things that didn't work and what worked.

Git commit ID or dataset versions can be recorded in the journal if needed. Or just keep the dates and make sure you push every day.

### Comments on stopword removal, stemming, lemmatization

Jeremy says it's a terrible idea. The rule of thumb is to leave the raw text alone and do not do the stopwords removal, stemming, etc. These are for traditional NLP before deep learning. We don't want to lose information.

(Me: some of these processing may still be useful for certain tasks. Jeremy's answer here works for text classification in general.)

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

## ULMFiT is transfer learning applied to AWD-LSTM

Jeremy: LSTMs and RNNs are **not** inferior to transformer models such as GPT-2 and BERT, and they should not be a thing in the past.

- Transformers and CNNs for texts don't have state. For example, for speech recognition, you need to do analyses for all the samples around one sample again and again, so they are super wasteful.
- They are fiddly, so they are not extensively used in industry-grade NLP yet. At the moment, Jeremy's go-to choice for real world NLP tasks is still ULMFiT and RNNs.

A language modeling is generic, it could be predicting the next sample of a piece of music or speech, or next genome in a sequence, etc.

### Text preprocessing

Notebook: `12_text`

#### 1. Make TextList

Adapt `ItemList` to `TextList`,

```py
#export
def read_file(fn):
    with open(fn, 'r', encoding = 'utf8') as f: return f.read()

class TextList(ItemList):
    @classmethod
    def from_files(cls, path, extensions='.txt', recurse=True, include=None, **kwargs):
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, i):
        if isinstance(i, Path): return read_file(i)
        return i
```

For any other custom task and files, just implement the `read_file` function, and override `get` in `ItemList`.

#### 2. Tokenization

Next, we use `spacy` for tokenization.

Before tokenization, we can write a list of preprocessing functions such as

- replace `<br />` with `\n`
- remove excessive spaces
- add space around `#` and `/`

or any custom behavior you want.

Then we define some symbols/tokens:

```py
UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = \
    "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()
```

Do custom things for them in text, and add the functions to a list and call it the `pre_rules`

```py
default_pre_rules = [
    fixup_text, replace_rep, replace_wrep, spec_add_spaces,
    rm_useless_spaces, sub_br
]
default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]
```

Jeremy finds `spacy`'s tokenizer complex but essential to produce good models. But it's slow. Use Python's `ProcessPoolExecutor` to speed it up.

```py
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor


def parallel(func, arr, max_workers=4):
    """Wrap ProcessPoolExecutor"""
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


class TokenizeProcessor(Processor):
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4):
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tokenizer = spacy.blank(lang).tokenizer
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i,chunk = args
        # Notice the use of `compose` to apply the pre_rules functions
        # in a functional programming way
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items):
        toks = []
        if isinstance(items[0], Path): items = [read_file(i) for i in items]
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        # Use `parallel` to speed up the tokenizer
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])

    def proc1(self, item): return self.proc_chunk([item])[0]

    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]
    def deproc1(self, tok):    return " ".join(tok)
```

#### 3. Numericalize tokens to vocab

Once we have tokenized our texts, we replace each token by an individual number, this is called numericalizing. Again, we do this with a processor (not so different from the `CategoryProcessor`).

```py
import collections

class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2):
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for o in reversed(default_spec_tok):
                if o in self.vocab: self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)})
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return [self.otoi[o] for o in item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]
```

#### 4. Batching

The way to do batch is to let each batch have different sequences of the same documents, because the RNN will have a state for each batch.

Each batch has size `(bs, bptt)`. This is really important.

Question: how to set `bs, bptt` combination? Jeremy says he doesn't know the answer, it's a good thing to experiment with.

To get rectangular tensor for texts of varying lengths, use padding tokens. This also works for rectangular images. Refer to the `pad_collate()` function.

### Build the RNN: AWD-LSTM

Notebook: `12a_awd_lstm`

Recall from part I, an RNN is just a regular neural network with as many layers as the number of tokens in the sequence to learn. For a long sequence we need so many layers, we use a for loop. Note that we use the same weight matrix for these layers (yellow in the diagram below).

<img src="{{ site.baseurl }}/images/fastai/basic-rnnn.png" alt="rnn" align="middle"/>

<img src="{{ site.baseurl }}/images/fastai/rnn.png" alt="rnn1" align="middle"/>

With say 2000 layers, we have problems like vanishing gradients and exploding gradients. To make things worse, we can have stacked RNNs with more thousands of layers.

To make the RNN easier to train, we use something called an LSTM cell.

<img src="{{ site.baseurl }}/images/fastai/lstm.png" alt="lstm" align="middle"/>

Conceptually this is a lot. The code is actually not much:

```py
class LSTMCell(nn.Module):
    def __init__(self, ni, nh):
        super().__init__()
        self.ih = nn.Linear(ni,4*nh)
        self.hh = nn.Linear(nh,4*nh)

    def forward(self, input, state):
        h,c = state
        #One big multiplication for all the gates is better than 4 smaller ones
        gates = (self.ih(input) + self.hh(h)).chunk(4, 1)
        ingate,forgetgate,outgate = map(torch.sigmoid, gates[:3])
        cellgate = gates[3].tanh()

        c = (forgetgate*c) + (ingate*cellgate)
        h = outgate * c.tanh()
        return h, (h,c)
```

There is another type of cell called GRU. They are both ways to forget things.

The fast option is to use pytorch cuda. There is also something called `jit` and it translates Python into C++ in the background. However it doesn't always work.

#### Dropout

Do dropout for an entire sequence at a time. Keywords: RNN dropout, weight dropout, and embedding dropout.

#### Gradient clipping

AWD-LSTM also uses gradient clipping to avoid exproding gradients. It allows us to use bigger learning rate.

### Train the language model

Notebook: `12b_lm_pretrain`, `12c_ulmfit`

Just use the code implemented in the previous two notebooks. It trains for ~5hrs to get the language model.

If a word is in the task dataset (e.g. IMDB) and isn't in the LM dataset (Wikitext103), we just use the mean weight and mean bias. If the word exists in both dataset, we directly use the embedding from the LM dataset.

Next, we create layer groups just as before, and train the model some more (~1hr), then we get a finetuned language model.

Tip: concat pooling is helpful for text as well as images.

## Papers

- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [Rethinking the Inception Architecture for Computer Vision ](https://arxiv.org/abs/1512.00567) (label smoothing is in part 7)
- [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187)
