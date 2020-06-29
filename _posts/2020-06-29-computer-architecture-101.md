---
toc: true
layout: post
description: CS foundamentals for non-CS majors
categories: [note, compsci]
title: "From NAND to TETRIS: Computer Architecture 101"
comments: true
---

Ever wondered how a computer executes your code with those tiny silicon thingys? If it sounds like a mystery to you, you need to read this article.

A computer system is a multiple-layer abstraction. You worry about *what* you want to do using *interfaces* that are *abstracted away* from you. Getting to know what the abstractions are and having an rough idea of how they work under the hood is an important and often overlooked prerequisite for a good coder. If you don't have this knowledge as a data scientist or software developer, you will feel you work on top of a floating and unreliable foundation.

In this article, I will introduce all the concepts in an easily consumable manner by walking through the process of building a general purpose computer called Hack from the ground up using first principles.

<img src="{{ site.baseurl }}/images/cs4ds/hack.png" alt="" align="middle"/>

After this exercise, you will walk away more confident and eliminate the insecurities of not knowing the foundational knowledge of computers.

Ready? Let's get started.

<img src="{{ site.baseurl }}/images/cs4ds/abstractions.png" alt="" align="middle"/>

## Level 1. Boolean operations and logic gates

<img src="{{ site.baseurl }}/images/cs4ds/booleanops.png" alt="" align="middle"/>

Booleans functions -> truth table.

<img src="{{ site.baseurl }}/images/cs4ds/truthtable.png" alt="" align="middle"/>

Boolean algebra laws

<img src="{{ site.baseurl }}/images/cs4ds/boollaws.png" alt="" align="middle"/>

How to construct a function from basic operations? (how to construct hardware)

Find all the 1s in truth table, write an expression for each, and OR them together! We can further simplify the expression. However, finding the shortest expression is NP hard.

> Any boolean function can be constructed with AND and NOT operations.

OR can be constructed with AND and NOT.

<img src="{{ site.baseurl }}/images/cs4ds/constructbool.png" alt="" align="middle"/>

### The amazing NAND operation

> Any boolean function can be constructed with NAND operations.

Because AND and NOT can be constructed with only the NAND gate.

<img src="{{ site.baseurl }}/images/cs4ds/nand.png" alt="" align="middle"/>

```
# Def: NAND
x NAND y = NOT(x AND y)

# NAND => NOT
NOT(x) = x NAND x

# NAND => AND
x AND y = NOT(x NAND y)
```

Logic gates are physical chips consisting of transistors that actually implement these boolean operations. We can use elementary logic gates such as AND, OR, NAND to construct composite gates.

Electrical engineers are responsible for constructing these gates with circuits and transistors. These chips have very clear specifications.

<img src="{{ site.baseurl }}/images/cs4ds/andtransistor.png" alt="" align="middle"/>
(wikipedia and gate)

We are not going to dive into the design of the circuit. We take the specifications of inputs and outputs and use them to build more complex logic. There is a type of language called Hardware Description Language (HDL) that can be used in this process. The two most popular HDLs are VHDL and Verilog, and there are others as well.

Designing composite gates from elementary ones takes experience. Here is an example of XOR built with AND, OR and NOT.

<img src="{{ site.baseurl }}/images/cs4ds/designxor.png" alt="" align="middle"/>

We can write HDL code and load it into a hardware simulator, run test scripts to make sure it works.

<img src="{{ site.baseurl }}/images/cs4ds/hardwaresim.png" alt="" align="middle"/>

A hardware construction project usually involves a system architect and some developers. The system architect breaks down the task into an overall chip API and smaller individual chips, and the developers build them using HDL.

### Multi-bit buses: an array of bits as one entity

When we manipulate many bits together, like adding two 16-bit integers, we can think of the bits as groups. Each group or array of bits is called "bus". HDL provides ways to manipulate buses as arrays.

<img src="{{ site.baseurl }}/images/cs4ds/hdlbus.png" alt="" align="middle"/>

These buses are indexed from right to left. Right means the least significant bit.

### Programmable elementary gates

The mutiplexor gate `mux` is an example of programmable gates. You give it a "selected bit" `sel` and it selects one of the inputs to be the output. This is a foundamental operations that exists in all kinds of programs.

<img src="{{ site.baseurl }}/images/cs4ds/mux.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/muxprogrammable.png" alt="" align="middle"/>

The demultiplexor gate `dmux` is the inverse of `mux`.

<img src="{{ site.baseurl }}/images/cs4ds/demux.png" alt="" align="middle"/>

A concrete example of an application using `mux` and `dmux` is a single communication line that interleaves two (or more) messages together!

<img src="{{ site.baseurl }}/images/cs4ds/muxdemux.png" alt="" align="middle"/>

## Level 2. Boolean arithmetic and ALU


## Level 3. Memory


## Level 4. Machine language


## Level 5. Computer architecture


## Level 6. Assembler



