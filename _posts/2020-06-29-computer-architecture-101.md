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

## Level 2. Boolean arithmetic and Arithmatic Logic Unit (ALU)

Doing binary addition is similar to decimal addition except that we carry over 1 to the next digit one position reaches 2 and not 10.

For an 8-bit binary number, we call "8" the **word size**. Anything outside the range of 8-bit representation is going to cause "overflow"

What the computer does when faced with overflow is just to ignore the overflow and truncate the result back to the word size.

<img src="{{ site.baseurl }}/images/cs4ds/overflow.png" alt="" align="middle"/>

### Building an adder

First we look at a "half adder". It takes in 2 bits and outputs a sum bit and a carry bit. The carry bit is 1 when the sum reaches 2.

<img src="{{ site.baseurl }}/images/cs4ds/halfadder.png" alt="" align="middle"/>

Then we can build a "full adder" that takes in 2 bits and a carry bit from a previous addition, and outputs a sum bit and a carry bit.

<img src="{{ site.baseurl }}/images/cs4ds/fulladder-diagram.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/fulladder.png" alt="" align="middle"/>

With a half adder for the least significant bit, and several full adders, we can build a multi-bit adder, say, for 16-bit binary numbers.

<img src="{{ site.baseurl }}/images/cs4ds/multibitadder.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/adder.png" alt="" align="middle"/>

### Negative numbers

We could use the first bit as the sign and the remaining bits for the actual number. But this approach has problems such as -0 and we need extra logic to handle subtraction.

The approach we actually use is called the 2's complement.

<img src="{{ site.baseurl }}/images/cs4ds/twoscomp.png" alt="" align="middle"/>

#### Negative addition

Negative addition comes for free because we throw away the overflowing bit, i.e. do a modulo $2^n$.

<img src="{{ site.baseurl }}/images/cs4ds/negadd.png" alt="" align="middle"/>

#### Negative subtraction: solve negation first

We use a math trick,

$$2^n - x = 1 + (2^n - 1) - x$$

We use it for these reasons:

1. Since $2^n - x = -x$ in our 2's complement representation, the left hand side is `-x`.
2. $(2^n - 1)$ in binary is `11111...1`, just `n` 1's, subtracting a binary number `x` from it is easy - just flip all its bits!
3. Now we have $(2^n - 1) - x$ in binary, adding 1 is trivial.

<img src="{{ site.baseurl }}/images/cs4ds/negsub.png" alt="" align="middle"/>

### Arithmatic Logic Unit (ALU)

The famous Von Neumann architecture

<img src="{{ site.baseurl }}/images/cs4ds/vonarch.png" alt="" align="middle"/>

ALU performs a function on input bits

<img src="{{ site.baseurl }}/images/cs4ds/alu.png" alt="" align="middle"/>

For the Hack computer we are building, we define the Hack ALU as follows,

<img src="{{ site.baseurl }}/images/cs4ds/hackalu.png" alt="" align="middle"/>

The pins at the top are the control bits.

The output of the ALU is determined completely by the truth table below,

<img src="{{ site.baseurl }}/images/cs4ds/hackalu-truthtable.png" alt="" align="middle"/>

There are many more functions we can compute, but we choose these ones for Hack.

The control bits are directives that ALU examines and **executes from left to right sequentially**.

- `if zx then x = 0`, zero all bits of x.
- `if nx then x != x`, flip all bits of x.
- `if zy then y = 0`, zero all bits of y.
- `if ny then y != y`, flip all bits of y.
- `if f then out = x + y, else out = x & y`
- `if no then out != out`, flip all bits of out.

You can check one operation, i.e. a row in the following truth table and verify that these sequential control directives output the right output for the desired operation on the inputs.

<img src="{{ site.baseurl }}/images/cs4ds/alu-example.png" alt="" align="middle"/>

We also have **output control bits** `zr` and `ng` defined as follows,

<img src="{{ site.baseurl }}/images/cs4ds/aluoutputcontrol.png" alt="" align="middle"/>

Why do we need these output control bits? The reason will become clear when we have the big picture of computer architecture.

The Hack ALU is simple, elegant and easy to implement.

> Leonardo da Vinci: Simplicity is the ultimate sophistication.

However, keep in mind that this is an extremely simplified version of the ALU. The real ALU in computers are much more complicated.

### Comments

#### Hardware / software tradeoff

There is a tradeoff between the hardware and software. For example, we can let the hardware do multiplication and division instead using software. A complex hardware is faster for operations but more costly to design and manufacture. It's the freedom of the hardware designer to decide whether to move some operations to software.

#### Unit testing

You might know that in software development we have unit tests. Actually, the idea comes from hardware development since we can isolate one logic unit and test whether it works correctly with its defined interfaces.

> Unit tests conceptually break a program into discrete pieces and test each piece in isolation, while integration tests make sure that the pieces are working properly together as a whole. Unit tests are generally small and don’t take much time to write because they only test a small section of code.



## Level 3. Memory


## Level 4. Machine language


## Level 5. Computer architecture


## Level 6. Assembler



