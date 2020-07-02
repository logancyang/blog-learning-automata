---
toc: true
layout: post
description: CS foundamentals for non-CS majors
categories: [note, compsci]
title: "From NAND to TETRIS: Computer Architecture 101. Part II"
comments: true
---

## Level 4: Machine language

Before we build the computer, let's understand what we want the computer to be able to do from a user point of view.

The reason that computers can do a lot of different tasks is because it is modeled based on Universal Turing Machine in theory, and Von Neumann Architecture in practice.

<img src="{{ site.baseurl }}/images/cs4ds/turing-von.png" alt="" align="middle"/>

<img src="{{ site.baseurl }}/images/cs4ds/vonneumann.png" alt="" align="middle"/>

The machine is designed to read binary instructions and perform different tasks. The binary instructions are called machine language. It is one of the most important elements of computer science because it is how software controls hardware.

Machine language consists of three conceptual parts:

1. Operations. It has a list of operations encoded in binary to go through. Each operation instructs it to do a different thing.
2. Program counter. Let the computer know which operation to execute next.
3. Address. Let the computer know what to operate on, where it can get it and where to put the output.

<img src="{{ site.baseurl }}/images/cs4ds/machinelang.png" alt="" align="middle"/>

We humans don't program in machine language. We write high level language and let the compiler to translate it to machine language. But since now we are learning how to build a computer, we need to understand machine language.

Machine language is just a sequence of bits. For example, the ADD operation is defined as `0100010`, say a variable R3 is `0011`, R2 is`0010`, then `010001000110010` means `ADD R3 R2`.

We allow humans to write machine language via Assemly language. In Level 6, we will build "Assembler" to convert it into bits.

Machine language is usually in close correspondence to actual hardware architecture. If we want more powerful machine language operations, we need more advanced hardware.

Common machine operations:

- Arithmetic
- Logical operations
- Flow control: goto instruction Y, if A then goto instruction C

Differences between machine languages:

- Richness of operations (division? bulk copy?)
- Data types (width, floating point...)

For example, some machines can only handle 8-bit arithmetic. To do a 64-bit addition, it can still do it by iterating through its 8 bits and relying on algorithm. But a 64-bit machine can do it much faster and more easily than the 8-bit machine.

### Memory hierarchy

Accessing a memory location is expensive.

- large memory has long addresses
- getting the memory content into CPU takes time (slow compared to CPU carrying out the operation)

The solution was introduced by Von Neumann when he built the first computer: memory hierarchy.

<img src="{{ site.baseurl }}/images/cs4ds/mem-hier.png" alt="" align="middle"/>

We have CPU registers (smallest and fastest), then cache, main memory, and disk (biggest and slowest). Smaller memory is faster because there are only a few registers, the addresses are short, and they sit closer to CPU.

#### CPU registers

CPU registers are built with the fastest technology. And since they sit inside the CPU, there's almost no delay.

There are 2 types of registers in the CPU.

- Data registers. We can put numbers in them directly.
- Pointer to main memory, e.g. put variable `X` into `@A` where `A` is an address in the main memory.

<img src="{{ site.baseurl }}/images/cs4ds/addressing-modes.png" alt="" align="middle"/>

### Input/Output

CPU needs some kind of protocal known as **drivers** to talk to input/output devices, such as mouse, keyboard, camera, sensors, screen, printer, sound, etc.

One general method of interaction uses "memory mapping", e.g.

- Memory location 12345 holds the direction of the last mouse movement.
- Memory location 45678 is not a real memory location but a way to tell the printer which paper to use.

### Flow control

Flow control is the way we tell the hardware which instruction to execute next.

There is unconditional jump so we can loop:

<img src="{{ site.baseurl }}/images/cs4ds/uncond-jump.png" alt="" align="middle"/>

There is conditional jump if a condition is met:

<img src="{{ site.baseurl }}/images/cs4ds/cond-jump.png" alt="" align="middle"/>

### The Hack computer and machine language

The Hack computer we are building is going to a 16-bit computer with an architecture shown below:

<img src="{{ site.baseurl }}/images/cs4ds/hack16bit.png" alt="" align="middle"/>

It has

- a data memory (RAM)
- an instruction memory (ROM)
- a CPU consists of 16-bit ALUs that performs 16-bit instructions
- some buses to move data between them. Think of them as highways of 16 lanes moving 16-bit data around.

#### Hack software

We design the Hack machine language to have 2 types of instructions:

- 16-bit A-instructions
- 16-bit C-instructions

**Hack program = a sequence of instructions written in the Hack machine language.**

The Hack program is loaded into the instruction memory (ROM), then the reset button is pushed to start the program. The reset button is pushed once for one program.

<img src="{{ site.baseurl }}/images/cs4ds/hackreset.png" alt="" align="middle"/>

The Hack machine language recognizes 3 registers:

- D register: holds a 16-bit value
- A register: holds a 16-bit value or an address
- M register: called the selected memory register, a 16-bit RAM register addressed by A.

<img src="{{ site.baseurl }}/images/cs4ds/reg3.png" alt="" align="middle"/>

Notice that ROM (which stores the instructions) isn't included in the 3 registers!

#### A-instruction

Example: `@100`

When this A-instruction is executed, the A register holds `100`, and `RAM[100]` is selected in the M register (RAM).

```
// Set RAM[100] to -1
@100    // A=100, select RAM[100]
M = -1  // RAM[100]=-1
```

The above Hack machine language means we select `RAM[100]` by setting A register to 100. `M` denotes the memory content of `RAM[100]`, we assign -1 to it.

#### C-instruction

C-instruction is the work horse of the language. It has 3 fields: `dest = comp ; jump`

1. Computation. Consists of logical operations on A, D and M.
2. Destination (optional)
3. A jump directive (optional)

<img src="{{ site.baseurl }}/images/cs4ds/cinst-ex1.png" alt="" align="middle"/>

Refer to the slide above for possible `comp`, `dest`, `jump` values.

Example: set RAM[300] to D-1

```
@300
M = D-1  // D-1 is in the predefined comp as shown above
```

Example: if D-1 == 0, jump to execute the instruction stored in ROM[56]

```
@56         // A=56
D-1; JEQ    // if D-1 == 0, goto 56
```

`JEQ` checks if `comp` equals 0, if yes then jump to address A (keep in mind that `jump` aims at ROM addresses, not RAM). In this case, `comp` is `D-1`, so `JEQ` checks if `D-1` equals 0.

Example code:

```
@1
M=A-1; JEQ
```

What does this do?

1. `@1` sets A register to 1.
2. Compute `comp` which is `A-1`: `A-1` equals 0.
3. Then it stores the result 0 into the M register, RAM[1], because A is 1.
4. `JEQ` checks if `comp` equals 0. Yes.
5. The next instruction is ROM[1] because A is 1.

We don't need to remember all the possible values for the 3 registers. For details, check the [website](https://www.nand2tetris.org/).

### Hack language specification


















