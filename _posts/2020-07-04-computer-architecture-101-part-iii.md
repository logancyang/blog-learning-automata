---
toc: true
layout: post
description: CS foundamentals for non-CS majors
categories: [note, compsci]
title: "[From NAND to TETRIS] Computer Architecture 101 Part III: Computer Architecture and Assembler"
comments: true
---

Continuing the discussion through abstraction level 4 in the [last post](http://blog.logancyang.com/note/compsci/2020/07/02/computer-architecture-101-part-ii.html), we look at level 5 and level 6 in this post.

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

Fetch:

- put the **location** of the next instruction (available from the program counter) into the **address** of the **program memory**
- and then get the instruction code itself by reading the memory content at that location.

<img src="{{ site.baseurl }}/images/cs4ds/fetch.png" alt="" align="middle"/>

In the execute cycle, we use the bits from the control bus and access the registers and/or the data memory to manipulate the data.

There is clash between fetch and execute because they access the program and data memory which are in the same physical memory. Usually, we use a multiplexor to do the fetch cycle first and remember the content of the instruction in the register, and then access the data memory. There's a variant of the Von Neumann Architecture which is the Harvard Architecture. It physically separate the program and data memory to avoid this complication.

### The Hack CPU

The hack CPU is the brain of the computer, it decides which instruction to execute next and how.

It has 3 inputs:

1. Data value
2. Instruction
3. Reset bit

<img src="{{ site.baseurl }}/images/cs4ds/hackcpuinput.png" alt="" align="middle"/>

And 4 outputs:

1. Data value
2. Write to memory (Y/N)
3. Memory address
4. Address of the next instruction

<img src="{{ site.baseurl }}/images/cs4ds/hackcpuoutput.png" alt="" align="middle"/>

Here's the Hack CPU architecture:

<img src="{{ site.baseurl }}/images/cs4ds/hackcpu.png" alt="" align="middle"/>

All the chips used are the ones we have built before. We just need to assemble them together in this way. The `C` bits are control bits.

Here is how the CPU recognizes and handles an A-instruction:

<img src="{{ site.baseurl }}/images/cs4ds/insthandling.png" alt="" align="middle"/>

Here is how the CPU recognizes and handles a C-instruction:

<img src="{{ site.baseurl }}/images/cs4ds/insthandling-c.png" alt="" align="middle"/>

Looking at ALU:

<img src="{{ site.baseurl }}/images/cs4ds/alu-inputs.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/alu-outputs.png" alt="" align="middle"/>

**Reset bit: program starts/restarts when reset is 1.**

Here is the control implementation and program counter logic:

<img src="{{ site.baseurl }}/images/cs4ds/cpucontrol.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/controlimpl.png" alt="" align="middle"/>

### Data memory

Recall the overall Hack architecture:

<img src="{{ site.baseurl }}/images/cs4ds/hackoverall.png" alt="" align="middle"/>

Now that we have the CPU implementation, we look at the memory. It has 3 parts: the RAM part that is the data memory, plus the Screen Memory Map, and the kernel memory map.

<img src="{{ site.baseurl }}/images/cs4ds/ram.png" alt="" align="middle"/>

The fist part, RAM16, is already implemented in our previous sections.

We also went through Screen Memory Map and Keyboard Memory Map in detail before. One special feature of the Screen chip is that it can refresh the display at a frequency.

### ROM32K: Instruction memory

<img src="{{ site.baseurl }}/images/cs4ds/rom.png" alt="" align="middle"/>

To put a program into ROM, there are 2 ways

- Burn the program onto a ROM chip, such as a CD, and run the program with the CD.
- Load the program which is written in a text file using **hardware simulation** by the built-in ROM chip.

### Putting it all together

<img src="{{ site.baseurl }}/images/cs4ds/arch.png" alt="" align="middle"/>

### Comments

- We can have a chip that tracks the state the computer is in to make sure it performs the right set of instructions given its state. We use the concept of Finite State Machine to organize the states and their transitions.
- To add more peripheral input/output devices, we can allocate more segments in the memory for them.

## Level 6. Assembler



## Reference

[Coursera From NAND To TETRIS](https://www.coursera.org/learn/build-a-computer)
