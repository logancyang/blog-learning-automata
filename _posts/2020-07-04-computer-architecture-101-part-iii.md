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

For the first time in this series, this part is software. The Assembler translates the machine language to binary code that can be executed. It is the first software layer above the hardware.

But where do we run the Assembler if it is not already in 0s and 1s itself?

Since we are not building the first computer in the world, we can write the Assembler on another computer, compile it down to binary, transfer the binary to the Hack computer we are building, and use the binary directly.

<img src="{{ site.baseurl }}/images/cs4ds/assembler.png" alt="" align="middle"/>

The Assembler reads a line of Hack code into an character array. It ignores comments and whitespaces when breaking down the line into fields.

The Assember also supports symbols (variable names). The way it supports that is to store the symbol name at an address.

<img src="{{ site.baseurl }}/images/cs4ds/symbol-addr.png" alt="" align="middle"/>

Essentially, a symbol is just an alias for an address location.

It is possible in the Assembly (Hack) machine language that a "jump" symbol is used before it is declared using the `@` operator. One solution is to do 2 passes for the code, the first pass builds the symbol table, and the second pass uses it to correctly executes the program.

<img src="{{ site.baseurl }}/images/cs4ds/symbols.png" alt="" align="middle"/>

### The translator perspective

We have already seen the tables that map A and C instructions to binary code. Along with user symbols that are constructed by the declarations, we have all we need to do the translation.

We take care of

- White spaces, empty line, indentation
- Comments, inline comments
- A, C instructions
- Symbols

#### Step 1: Handling whitespaces

Let's focus on them one by one. First we handle whitespaces for a simplified Hack program with no symbols.

This is the easiest one. We simply **strip all whitespaces, so the program after this step has no whitespace**.

#### Step 2: Translating A-instructions

An A-instruction is just `@value`. In binary, we prepend the op code `0` for A-instruction, and convert the value in decimal to binary, add 0 paddings accordingly.

<img src="{{ site.baseurl }}/images/cs4ds/trans-a.png" alt="" align="middle"/>

#### Step 3: Translating C-instructions

Consider `MD=D+1`. This C-instruction has a `dest: MD` and `comp: D+1`, with no `jump` field.

The binary syntax for C-instructions is

```
1 1 1 a c1 c2 c3 c4 c5 c6 d1 d2 d3 j1 j2 j3
```

3 1s at the beginning, and fill in the `a, c, d, j` bits according to the table below

<img src="{{ site.baseurl }}/images/cs4ds/trans-c.png" alt="" align="middle"/>

#### Step 4: Handling symbols

<img src="{{ site.baseurl }}/images/cs4ds/handle-symbol.png" alt="" align="middle"/>

For pre-defined symbols, we can simply replace them with their values based on the table below

<img src="{{ site.baseurl }}/images/cs4ds/pre-symbols.png" alt="" align="middle"/>

For the label symbols, they are used for `jump` commands.

- The program uses the label to "goto" another instruction
- Declared with `()`
- A label symbol points to the memory location of its next instruction. `label -> mem_index_of_next_instruction`

When translating a label symbol, simply replace it with the value `mem_index_of_next_instruction`.

For the variable symbols, they are assigned to memory locations starting from address 16 (a designer's decision).

We need a **symbol table** to keep track of the new and old variable symbols. It has key-value pairs of symbol name and symbol address.

The symbol table is initialized with all the predefined symbols and their addresses.

The Assembler performs **2 passes**. In the first pass adds the **label symbols** to the symbol table. The second pass adds the variable symbols to the symbol table.

The symbol table is an auxilary data structure that the Assembler needs to process the program. Once the processing completes, the symbol table can be tossed away.


### Overall Assembler logic

Keep in mind, the Assembler is just a **text processor** that takes in the Assembly/Hack language and returns a file containing only two types of characters, 0 and 1. These 0s and 1s are still ASCII characters. When the computer loads it into memory it then becomes *real* 0s and 1s.

> Although we say we are not building the world's first computer and resort to a high level language to write this Assembler text processor, all that matters is the resulting translation directive text file. One *can* write it from scratch with hardware logic, but it's cumbersome. It's not that we are not able to produce this text processor without the help of a high level language. We already have all the logic we need.

<img src="{{ site.baseurl }}/images/cs4ds/assembler-overall.png" alt="" align="middle"/>

The overall Assembler process:

<img src="{{ site.baseurl }}/images/cs4ds/assembler-proc.png" alt="" align="middle"/>

To implement the Assembler in a high level language like Python or Java, here's the pseudocode:

<img src="{{ site.baseurl }}/images/cs4ds/assembler-pseudo.png" alt="" align="middle"/>



### Macro assembler and macro commands

We can make the Hack language more user friendly by adding *macro commands*. For example, we can convert

```
@100
D=M
```

to

```
D=M[100]
```

To achieve this, simply add the translation logic into the Assembler.

### How was the first Assembler created in history without the assistance of high level languages?

The first Assembler was built **by hand**. People write the binary code for the translation process. However, we only needed to do it **once**. Once it's created, people can build on it and create new Assemblers by programming.

## From NAND to TETRIS Part I: Last words

Now we have built the entire Hack computer from NAND gates!

Outside the classroom, it is rare that people need to work with the Assemly language. People only work with high level languages in programming unless they really need to optimize a piece of code for a specialized device.

The most important concept introduced here is that **why computers are programmable** and can do all kinds of tasks.

**We bridged the gap from hardware to software**. This is the key idea we learned!

In *[From NAND to TETRIS Part II](https://www.coursera.org/learn/nand2tetris2)*, the material will cover the software layers such as the virtual machine, the compiler, operating system, etc.

## Reference

- [Coursera From NAND To TETRIS](https://www.coursera.org/learn/build-a-computer)
- The amazing lecturers:
  - [Shimon Schocken](https://www.coursera.org/instructor/shimon)
  - [Noam Nisan](https://www.coursera.org/instructor/noamnisan)
