---
toc: true
layout: post
description: CS foundamentals for non-CS majors
categories: [note, compsci]
title: "[From NAND to TETRIS] Computer Architecture 101 Part III: Computer Architecture"
comments: true
---

Continuing the discussion through abstraction level 4 in the [last post](http://blog.logancyang.com/note/compsci/2020/07/02/computer-architecture-101-part-ii.html), we look at level 5: computer architecture in detail in this post.

## Level 5: Computer Architecture

We are going to build the Von Neumann Architecture that is capable of executing any program.

<img src="{{ site.baseurl }}/images/cs4ds/vonneumann.png" alt="" align="middle"/>

### Von Neumann Architecture

<img src="{{ site.baseurl }}/images/cs4ds/infoflow.png" alt="" align="middle"/>

The CPU has ALUs and registers. The ALUs do the arithmetic and logical operations.

The memory holds the data and the program.

To move data around, there are 3 kinds of buses: the data bus, address bus, and control bus (or control wires).

**The ALU loads information from the Data bus and manipulates it using the Control bits.**

We need a two-way connection between the CPU registers and the data bus for intermediate results. We also need to connect CPU registers to the address bus (one way) because we access the memory by the memory address in the registers. So **the CPU registers have both data and addresses**.

For the memory, we need to connect the data memory to the data bus, and the program memory to the control bus.

### The Fetch-Execute Cycle

All that the CPU does is fetch an instruction and execute it, and repeat. This is the CPU loop.









## Reference

[Coursera From NAND To TETRIS](https://www.coursera.org/learn/build-a-computer)
