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

## Level 2. Boolean arithmetic and Arithmetic Logic Unit (ALU)

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

### Arithmetic Logic Unit (ALU)

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

### Time

Now we will learn how the computer executes one thing after another, i.e. how it perceives **time**.

The way we let a computer perceive time is to use some kind of digital oscillator as a discrete time clock. This way we convert physical time into discrete time.

Since the hardware uses voltage to represent bit state 0 and 1, and it takes time for the hardware to stablize after a state transition, we design the unit time step in the discrete clock to be slightly bigger than the time it takes to stablize. We then sample the stable state to be the state of that time unit.

A time unit is at the atomic level, **the computer only knows one state within one time unit**.

<img src="{{ site.baseurl }}/images/cs4ds/clock.png" alt="" align="middle"/>

### Combinatorial vs. sequential logic

Combinatorial logic just means that the output for a time unit only depends on the input in that time unit.

```
out[t] = function(in[t])
```

Sequential logic means we **use the same wire to store the bit**, and current step's output depends on the last time step.

```
state[t] = function(state[t-1])
```

This is the prototype of the **iterator**. With the sequential logic we can operate with time as input to our function.

<img src="{{ site.baseurl }}/images/cs4ds/sequential.png" alt="" align="middle"/>

### Flip Flops: Hardware that implements the sequential logic

Previously we had logic gates and ALUs that can compute a variety of operations, but we are missing one key piece to enable sequential logic: the hardware to remember the state from time t-1 to time t. How to do that?

We need something called the Clocked Data Flip Flop. Its output is input shifted one step forward.

<img src="{{ site.baseurl }}/images/cs4ds/dff.png" alt="" align="middle"/>

The little triangle in the diagram means it has a time-dependent chip. The combinatorial chips get the outputs instantaneously, while the sequential logic chips are time-dependent.

Conceptually, with the NAND gate and the Data Flip Flop as a foundation, we can build everything needed in a computer!

We are not going to describe how the flip flops are built physically here. The digital circuit is quite clever and elegant though. If you are interested, read about it [here](https://en.wikipedia.org/wiki/Flip-flop_(electronics)#D_flip-flop).

### 1-bit register

We can build a chip by using the D Flip Flop that remembers the last input state when the load state is 1.

<img src="{{ site.baseurl }}/images/cs4ds/onebitreg.png" alt="" align="middle"/>

The logic is:

```
if load(t-1):
    out[t] = in[t-1]
else:
    out[t] = out[t-1]
```

A load of 1 at t-1 means we need to remember the input at t-1 from now on until a next load of 1 regardless of how input changes during this period.

How to implement a 1-bit register using the D Flip Flop? Recall the **mutiplexor** (select one of two inputs based on a "select" bit).

<img src="{{ site.baseurl }}/images/cs4ds/onebitreg-mux.png" alt="" align="middle"/>

With the 1-bit register, we can start to build the memory unit!

### The Memory Unit

There are different types of memory. The most important type is the RAM. **It stores both data and instructions**.

It is a **volatile device** meaning it depends on an external power supply to store information. Physically it is not like the disk or ROM (read-only memory), it clears everything whenever the computer is disconnected from power while in the disk or ROM, info is persisted.

<img src="{{ site.baseurl }}/images/cs4ds/memorypic.png" alt="" align="middle"/>

We can build an N-bit register by putting N 1-bit registers side by side. Here we talk about 16-bit registers without loss of generality. The register's **state** is the value stored in the register. To lookup the state, we just need to probe its output pins.

To store a new input state `v`, we set `load` to 1 at that time step.

**The RAM abstraction: A sequence of *n* addressable registers with addresses 0 to n-1.**

**Important: At any given time, ONLY ONE REGISTER in the RAM is selected!**

Since we have *n* addressable registers, we need a binary number to represent the address. Turning *n* to a binary number, we say it has *k* bits. So `k = log2(n)`.

<img src="{{ site.baseurl }}/images/cs4ds/memoryparams.png" alt="" align="middle"/>

**RAM is a sequential chip with a clocked behavior.**

> Why is it called Random Access Memory?

> A: Because regardless of how many registers the memory unit has, be it 1 million or 1 billion, it takes exactly the same access time to access any one of the registers with a given address!

Some different RAM chips:

<img src="{{ site.baseurl }}/images/cs4ds/ramchips.png" alt="" align="middle"/>

### The Counter

A Program Counter (PC) is a chip (hardware device) that realizes the following 3 abstractions:

1. Reset: fetch the first instruction, PC = 0
2. Next: fetch the next instruction, PC++
3. Goto: fetch instruction n, PC = n

Here is the logic the Counter does:

```
if reset[t] == 1:
    out[t+1] = 0  # reset counter to 0
elif load[t] == 1:
    out[t+1] = in[t]  # set counter to input value at t
elif inc[t] == 1:
    out[t+1] = out[t] + 1  # increment counter by 1
else:
    out[t+1] = out[t]  # maintain the last value
```

<img src="{{ site.baseurl }}/images/cs4ds/counter.png" alt="" align="middle"/>

### ROM vs. RAM vs. Flash vs. Cache memory

The ROM (read-only memory) is an involatile device that keeps information persisted even without external power supply. For example, it is used in the computer booting process for this reason.

Another technology is flash memory. It combines the good things of both ROM and RAM, it doesn't need external power supply to persist data, and is writable.

Cache memory is a small and expensive memory that is close to the processor. It has very high performance. There is a hierarchy of cache memories, the closer to the processor, the faster, smaller and more expensive it gets.

## Level 4. Machine language


## Level 5. Computer architecture


## Level 6. Assembler



