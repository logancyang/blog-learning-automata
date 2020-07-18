---
toc: true
layout: post
description: Stepping up JavaScript skills
categories: [note, p5js, javascript]
title: "[JavaScript Essentials] Prototype and Inheritance"
comments: true
---

Not like Java or Python, JavaScript is based on *prototypes* and not *class*. The ES6 class syntax is syntactic sugar on top of prototypes. Here let's take a look at the prototype with some examples.

In ES5 we use a function as the constructor.

```js
function Particle() {
    this.x = 100;
    this.y = 99;
}
```

When you say `new Particle()`, JavaScript knows that the function is a constructor and will execute it as such.

The idea of Object Prototype is that you can define things in the prototype that all objects can share.

```js
p1 = new Particle();
p2 = new Particle();

// Define a function `show` that works for all Particles
Particle.prototype.show = function() {
    // `point` is a p5 function
    point(this.x, this.y);
}

// Now we can call
p1.show();
```

The way it works is that when you call `p1.show()`, JavaScript first tries to find `show()` in `p1`'s definition, if it's not there, it tries the `Particle` definition. `Particle` has a property `__proto__` that points to `Object.prototype` that has all the properties `Particle` objects share.

**Every JavaScript object has `Object.prototype`, such as arrays, functions, etc.**

`Object.prototype` has some special properties. One is `hasOwnProperty()`.

```js
p1.hasOwnProperty('z');
// false
p1.hasOwnProperty('x');
// true
```

One interesting thing about `hasOwnProperty()` is that it doesn't go up the *prototype chain*, it only finds the property of that instance.

```js
p1.hasOwnProperty('show')
// false
```

## Prototype inheritance chain

Consider this example where `Confetti` should inherit from `Particle` and have its own `show` function.

```js
function Particle() {
    this.x = 100;
    this.y = 99;
}

function confetti() {
    // Have the Particle constructor
    Particle.call(this);
    this.color = color(255, 0, 255);  // Pink
}

Particle.prototype.update = function() {
    // random is from p5
    this.x += random(-5, 5);
    this.y += random(-5, 5);
}

Particle.prototype.show = function() {
    stroke(255);
    strokeWeight(8);
    point(this.x, this.y);
}

// Create a new object as Confetti.prototype
// It is a copy of Particle.prototype
Confetti.prototype = Object.create(Particle.prototype);
Confetti.prototype.constructor = Confetti;

Confetti.prototype.show = function() {
    stroke(this.color);
    noFill();
    strokeWeight(8);
    square(this.x, this.y, 50);
}
```

The key here is that we need to create a copy of `Particle.prototype` and assign it to `Confetti.prototype` so that when we modify the latter it won't affect the former. This is how inheritance works under the hood in JavaScript.

But doesn't this look bad? Good news is that in ES6 we can use **class** for inheritance instead of this!

For ES6 inheritance, refer to this post: [[Nature of Code] Part I: Creating Physics Engine](http://blog.logancyang.com/note/natureofcode/simulation/processing/p5js/2020/03/13/nature-of-code-1.html)

## Reference

- The Coding Train
  - <https://youtu.be/hS_WqkyUah8>
  - <https://youtu.be/CpmE5twq1h0>
