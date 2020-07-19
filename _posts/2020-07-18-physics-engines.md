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

## Reference

- The Coding Train [videos](https://youtu.be/urR596FsU68)
