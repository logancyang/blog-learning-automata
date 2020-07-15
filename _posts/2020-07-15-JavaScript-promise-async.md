---
toc: true
layout: post
description: Stepping up JavaScript skills
categories: [note, p5js, javascript]
title: "[JavaScript Essentials] Promise, fetch, async and await"
comments: true
---

## Promises (ES6)

A callback function is associated with an **event**, e.g. when the mouse is pressed, trigger some function.

**Example task: Consider a series of API calls for data, say you call `wordnikAPI` to get a word, *and then* use that word to call `giphyAPI` to get a gif for it.**

You then need to nest callback in callback. If you have a long chain of calls, it can quickly become super nested and unwieldy. This is called **callback hell**.

```js
// p5.js loadJSON
loadJSON(url, function(data) {
    loadJSON(url, function(data) {
        loadJSON(url, function(data) { ... });
    });
});
```

Instead of passing a callback function, we can use a `Promise` object,

```js
// fetch() returns a Promise. It is native JavaScript
let promise = fetch(url);
```

It can have one of several states: `pending, fulfilled, rejected`.

With the promise object defined, we can chain `then()` and `catch()` to it.

**`then()` is for the state `fulfilled`, `catch()` is for the state `rejected`**.

The usual way to use `fetch` is chaining.

**Keep in mind that `fetch` returns a promise that needs `.json()` to convert it into something we can use. But the object returned by `.json()` is also a promise!**

```js
fetch(url)
  .then(resp => resp.json())
  .then(json => {/* do something with json data */})
  .catch(err => {/* do something with err */});
```

To do 2 API calls one after the other, and use the result of the first for the second call, with promise chaining:

```js
fetch(url1)
  .then(resp => resp.json())
  .then(json => {
    /* do something with json data */
    return fetch(url2_with_json_data)
  })
  .then(resp => resp.json())
  .then(json => {/* do something with json data from url2 */})
  .catch(err => {console.error(err)});
```

In every step of this chain, there could be an error. The final `catch` can catch all of them. This is much nicer than writing a callback to handle the error for each step.

**Don't forget to *return* the promise in the middle of the chain!**

### Make your own `Promise`

Consider this callback example

```js
setTimeout(sayHello, 1000);

function sayHello() {
  createP("Hello!");
}
```

`Hello!` will pop up in the webpage 1 second after loading the page.

What if we want to use a Promise? We need to create pathways to resolve the promise.

```js
delay(1000)
  .then(() => createP("Hello"))
  .catch(err => console.error(err))

// `delay` need to return a new Promise
function delay(time) {
  return new Promise((resolve, reject) => {
    if (isNaN(time)) {
      // reject promise. Error will be caught by .catch()
      reject(new Error('delay requires a number'));
    } else {
      // This means "after this amount of time, resolve the promise"
      setTimeout(resolve, time);
    }
  });
}
```

## `async/await` keywords (ES8)

`async` and `await` are syntactic sugar for Promises. It makes things easier to read, but it doesn't provide extra functionalities.

Let's consider the previous example where we need to chain a lot of `then`s together.

```js
fetch(url1)
  .then(resp => resp.json())
  .then(json => {
    /* do something with json data */
    return fetch(url2_with_json_data)
  })
  .then(resp => resp.json())
  .then(json => {/* do something with json data from url2 */})
  .catch(err => {console.error(err)});
}
```

This looks bad. What we can do instead:

```js
async function wordGIF() {
  const data1 = await fetch(url1);
  const json1 = data1.json();
  const word = json1.word;
  // It is fine to console.log the data here, it resolved
  console.log(word);
  // We can await multiple fetches!
  const data2 = await fetch(url2);
  const json2 = data2.json();
  const image_url = json2.data[0].images['fixed_height_small'].url;
  // Returns promise to be resolved
  return {
    word,
    image_url
  }
}

wordGIF().then(results => {
  createP(results.word);
  createImg(results.img_url);
}).catch(err => console.log(err));
```

## Promise.all()

Say `wordGIF(num)` takes a number and it gets a word of that length from wordnik API. Let's look at a comparison

```js
wordGIF(3).then(results => {
  createP(results.word);
  createImg(results.img_url);
  return wordGIF(4)
}).then(results => {
  createP(results.word);
  createImg(results.img_url);
})
.catch(err => console.log(err));

// vs.

wordGIF(3).then(results => {
  createP(results.word);
  createImg(results.img_url);
}).catch(err => console.log(err));

wordGIF(4).then(results => {
  createP(results.word);
  createImg(results.img_url);
}).catch(err => console.log(err));
```

The top one has order, it runs `wordGIF(3)` first and then do `wordGIF(4)`.

The bottom one is parallel, there is no guaranteed ordering. `wordGIF(4)` may come back first and its word and image may get rendered above `wordGIF(3)`.

Instead of chaining them with `then()`, there is another way to maintain order: using `Promise.all()`.

```js
let promises = [wordGIF(3), wordGIF(4), wordGIF(5)];

Promise.all(promises).then(results => {
  for (const result of results) {
    createP(result);
    createImg(result);
  }
}).catch(err => console.error(err));
```

`Promise.all()` gets the array of results when all of them resolve, and returns them *in one batch*. It maintains the order, and the effect here is that they render at the same time.

The downside is that it's all or nothing, if any of them gets an error, it fails everything.

## `try`, `catch` with Promises

Since `Promise.all()` is all or nothing, we need to catch and handle the error for each case in the array if there is any.

```js
async function wordGIF(num) {
  const data = await fetch(url);
  const json = data.json();
  const word = json.word;

  let image_url = null;
  try {
    image_url = json2.data[0].images['fixed_height_small'].url;
  } catch (err) {
    console.log("No image found for " + word);
    console.error(err);
  }

  // Returns promise to be resolved
  return {
    word,
    image_url
  }
}

let promises = [];
for (let i = 3; i < 10; i++) {
  promises.push(wordGIF(i));
}

Promise.all(promises).then(results => {
  for (const result of results) {
    createP(result);
    if (result.image_url !== null) {
      createImg(result);
    }
  }
}).catch(err => console.error(err));
```

This will show the gifs that are successfully found and log the error message for those are not found!

## Next steps

- Learn to get promises in an array in concurrently, render as they come in.
- Add a loading bar
