---
toc: true
layout: post
description: Nature of Code note series
categories: [note, natureofcode, simulation, processing, p5js]
title: "[Nature of Code] Part 1.2: Physics Engines"
comments: true
---

This is the second note in the Nature of Code series. For how to create your own mini physics engine, checkout the previous [post](http://blog.logancyang.com/note/natureofcode/simulation/processing/p5js/2020/03/13/nature-of-code-1.html).

## Physics Engines

In the last note, we covered how to create a physics engine. Usually we just want to use it. There are several available physics engines that we can use out of the box.

- Box2D: famous, behind a lot of 2D mobile games like Angry Birds (2D)
- toxiclibs: Java library (2D and 3D)
- bullet (3D)
- matter.js (2D)
- canon.js (3D)
etc.

All these physics engines have the same paradigm: **world**, **body** (or shape), and **constraint** (or connection, or joint).

### When to use a physics engine

Use a physics engine instead of your own code when you have

1. collisions
2. a connected system: a cloth, a soft body, an elastic pendulum

Different physics engines have different advantages:

- Box2D: good at collisions
- toxiclibs: very good at connected systems, springs, it has no collision!
- matter.js: has both collision and connected system

If your scenario is simple, such as some bouncing balls, you can use your own implementation.

I will focus on **matter.js** for in-browser simulations.

## Introduction to Matter.js

To use matter.js, go to their [Get Started page](https://github.com/liabru/matter-js/wiki/Getting-started), download a stable release and put the `matter.js` under `build` into your project, then add the following to your HTML.

```html
<script src="path_to/matter.js"></script>
```

We will use P5.js as the renderer.

**IMPORTANT: Need to match the way matter.js and P5.js draw stuff. For example, matter.js has `rectangle` with `x, y` representing the center, and P5.js has `x, y` representing the top left corner! In this case, use `rectMode(CENTER)` in P5.js.**

One weird thing that could happen is, if you have a very thin ground or boundary, the objects can sometimes pass through it when they move very fast. This is because matter.js checks the position of objects in every time step. Fast moving objects can fly through a boundary without overlapping with it. One way to fix this is to have thicker boundaries.

Another common pitfall with physics engine is that you can't assign a body's property such as angle directly, you can only set it at the time of instantiation. Because angle is a result of the laws of physics after you create the object. When you directly assign it, you violate the law of physics.

Here's an example that lets you create dropping circles by dragging the mouse.

```js
class Circle {
  constructor(x, y, r) {
    const options = {
      friction: 0,
      restitution: 0.95
    };
    this.body = Bodies.circle(x, y, r, options);
    this.r = r;
    World.add(world, this.body);
  }

  show() {
    const pos = this.body.position;
    const angle = this.body.angle;
    push();
    translate(pos.x, pos.y);
    rotate(angle);
    rectMode(CENTER);
    strokeWeight(1);
    stroke(255);
    fill(127);
    ellipse(0, 0, this.r * 2);
    pop();
  }
}

class Boundary {
  constructor(x, y, w, h, a) {
    const options = {
      friction: 0,
      restitution: 0.95,
      angle: a,
      isStatic: true
    };
    this.body = Bodies.rectangle(x, y, w, h, options);
    this.w = w;
    this.h = h;
    World.add(world, this.body);
  }

  show() {
    const pos = this.body.position;
    const angle = this.body.angle;
    push();
    translate(pos.x, pos.y);
    rotate(angle);
    rectMode(CENTER);
    strokeWeight(1);
    noStroke();
    fill(0);
    rect(0, 0, this.w, this.h);
    pop();
  }
}

// sketch.js

const Engine = Matter.Engine;
const World = Matter.World;
const Bodies = Matter.Bodies;

let engine;
let world;
const circles = [];
const boundaries = [];

let ground;

function setup() {
  createCanvas(400, 400);
  engine = Engine.create();
  world = engine.world;

  boundaries.push(new Boundary(150, 100, width * 0.6, 20, 0.3));
  boundaries.push(new Boundary(250, 300, width * 0.6, 20, -0.3));
}

function mouseDragged() {
  circles.push(new Circle(mouseX, mouseY, random(5, 10)));
}

function draw() {
  background(51);
  // Notice this line, Engine.update(engine) within draw loop
  Engine.update(engine);
  for (const circle of circles) {
    circle.show();
  }
  for (const boundary of boundaries) {
    boundary.show();
  }
}
```

Check out this [video](https://youtu.be/uITcoKpbQq4) for the effect.

### Constraint

A `Constraint` is a connection between 2 bodies.

To use it:

```js
const Constraint = Matter.Constraint;
// ... define particle and prev_particle ...

const options = {
    bodyA: particle.body,
    bodyB: prev_particle.body,
    length: 20,
    stiffness: 0.4
};
const constraint = Constraint.create(options);
World.add(world, constraint);
```

`length` is the resting length of the "spring", stiffness ranges from 0 to 1, with 1 being very stiff.

### MouseConstraint

A `MouseConstraint` is a constraint between the mouse position (drag) and the body it clicked on.

The body a MouseConstraint clicked on is in a property `body` of its instance, `mouseConstraint.body`.

To use it:

```js
const MouseConstraint = Matter.MouseConstraint;
// ...

const canvasmouse = Mouse.create(canvas.elt);
// IMPORTANT: pixelDensity() gets the pixel density
// of your display. If this is not set correctly mConstraint won't work
canvasmouse.pixelRatio = pixelDensity();
const options = {
    mouse: canvasmouse
};
mConstraint = MouseConstraint.create(engine, options);
World.add(world, mConstraint);
```

Notice that we have to set `pixelRatio` correctly for it to work!

## Reference

- The Coding Train [videos](https://youtu.be/urR596FsU68)
