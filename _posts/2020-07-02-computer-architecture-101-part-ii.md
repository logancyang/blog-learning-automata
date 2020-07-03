---
toc: true
layout: post
description: CS foundamentals for non-CS majors
categories: [note, compsci]
title: "[From NAND to TETRIS] Computer Architecture 101 Part II: Machine Language"
comments: true
---

Continuing the discussion through abstraction level 1-3 in the [last post](http://blog.logancyang.com/note/compsci/2020/06/29/computer-architecture-101.html), we look at level 4: machine language in detail in this post.

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

CPU needs some kind of protocol known as **drivers** to talk to input/output devices, such as mouse, keyboard, camera, sensors, screen, printer, sound, etc.

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

**Notice that ROM (which stores the instructions) isn't included in the 3 registers!**

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

We have two ways of specifying machine language code, one in symbols and one in binary numbers. An *assembler* can translate symbolic code into binary code.

#### A-instruction specification

<img src="{{ site.baseurl }}/images/cs4ds/ainst-binary.png" alt="" align="middle"/>

A-instruction has an "op code" (first bit in binary code) of 0 at the beginning of the binary code.

#### C-instruction specification

C-instruction has an "op code" of 1, followed by 2 bits we don't use. By convention we set them to 1.

Here is a table where we can convert symbols to binary codes.

<img src="{{ site.baseurl }}/images/cs4ds/cinst-table.png" alt="" align="middle"/>

For example, if we want to convert `D+1`, we find the symbol, it's in the column `a=0`, and has a value of `011111`.

So the `comp` field `D+1` is `1110011111`.

We also have a table for `dest` the destination field.

<img src="{{ site.baseurl }}/images/cs4ds/cinst-dest-table.png" alt="" align="middle"/>

Similarly, we have a table for `jump` values. Altogether we have everything in one slide here:

<img src="{{ site.baseurl }}/images/cs4ds/cinst-table-full.png" alt="" align="middle"/>

Finally, here is an example of a small Hack program and its translation to binary.

<img src="{{ site.baseurl }}/images/cs4ds/hacklang-ex.png" alt="" align="middle"/>

### Working with registers and memory using machine language

#### Some examples

```
// D=10
@10     // There is no directive to directly set D=10. Set A=10 first
D=A     // and set D=A

// D++
D=D+1   // This is easy

// D = RAM[17]
@17
D=M

// RAM[17] = D
@17
M=D

// RAM[17] = 10
@10
D=A
@17
M=D

// RAM[5] = RAM[3]
@3
D=M
@5
M=D
```

Here's another example:

<img src="{{ site.baseurl }}/images/cs4ds/hackcode-ex.png" alt="" align="middle"/>

Note that the white spaces are ignored, and each line of code has a line number in the background automatically.

#### How to terminate the program properly

We haven't talk about program termination. If we execute our code naively, a malicious hacker could add malicious code after our code to do bad things. It is called *[NOP slide](https://en.wikipedia.org/wiki/NOP_slide#:~:text=In%20computer%20security%2C%20a%20NOP,address%20anywhere%20on%20the%20slide.)*, meaning they added null operations after the actual code and before the malicious code to hide the latter.

We use the fact that the computer never stands still and always executes something, **we end the program with an infinite loop** so it is under our control. This is the best practice for program termination.

```
...
@6
0; JMP
```

#### Built-in symbols

The Hack assembly language features *built-in symbols* which are virtual registers, not real registers. The symbols are `R0, R1, R2, ... , R15` and they correspond to values `0, 1, 2, ..., 15`.

Why do we need these virtual registers?

It is purely for **style and readability**.

For example,

```
// RAM[5]=15
@15
D=A

@5
M=D
```

The two `@x` lines are intended to do completely different things. The first is to set A and then set D. The second is to get address 5 in RAM and store D there.

It means when we see `@x` we don't know what it wants to do until we see the next line of code. So we introduce `R5`,

```
// RAM[5]=15
@15
D=A

@R5
M=D
```

When we use `@R5` it's exactly the same as `@5`, but it means finding the address 5 so that it's more readable for people.

Here are all the built-in symbols:

<img src="{{ site.baseurl }}/images/cs4ds/builtin-symbols.png" alt="" align="middle"/>

### Branching, variables, iteration using machine language

#### Branching

In any high level language there are many branching mechanisms such as if-else, while, switch, etc. In machine language, there is only one: *goto* using the `jump` directives.

Example:

```
// Program: signum.asm
// Computes:
//   if R0 > 0:
//       R1 = 1
//   else:
//       R1 = 0

@R0
D=M  // D = RAM[0]

@8
D; JGT  // if R0 > 0, goto 8

@R1
M=0  // RAM[1] = 0
@10
0; JMP  // end of program

@R1
M=1  // R1=1

@10
0; JMP  // end of program
```

You can see this code is quite unreadable if there's no comment or documentation. One thing we introduce to make it more readable is **labels**. One example is shown below. `(POSITIVE)` is a label that points to its next line. `@POSITIVE` is using the label to *goto* that line. This way we have a much more readable branching mechanism.

<img src="{{ site.baseurl }}/images/cs4ds/goto-labels.png" alt="" align="middle"/>

#### Variables

Say we want to exchange the values of R0 and R1:

```
// Program: flip.asm
// Flips the value of RAM[0] and RAM[1]

// temp = R1
// R1 = R0
// R0 = temp
```

The people who created the Assembler can define a contract: `@somevar` with no label `(somevar)` means it's a variable, and it will use `RAM[program_base_address+16]`. Any new variable will increment 16, i.e. use `+17`, `+18`, etc.

<img src="{{ site.baseurl }}/images/cs4ds/variables.png" alt="" align="middle"/>

#### Iterations

Suppose we want to compute `1+2+...+n`, we need an accumulator variable and an iteration in a high level language. The machine code is shown below:

<img src="{{ site.baseurl }}/images/cs4ds/iterations.png" alt="" align="middle"/>

### Pointers

From the machine's perspective, an array is just a block of memory that starts at a certain base memory location with a certain length. The following example is to set -1 into the array.

<img src="{{ site.baseurl }}/images/cs4ds/array-1.png" alt="" align="middle"/>

First we set 2 variables, the base memory location `arr` and length `n`. Then we set the variable `i`. To achieve `RAM[arr+i] = -1`, we set register A to be `D+M`. This is the first time we set A using an arithmetic operation.

<img src="{{ site.baseurl }}/images/cs4ds/pointers.png" alt="" align="middle"/>

**Variables that store memory addresses like `arr` and `i` are called *pointers***. When we need to access memory using a pointer, we need an instruction like `A=M`.

Typical pointer semantics: "set the address register to the content of some memory register".

### Input/Output using machine language

The computer gets data from humans via input devices like the keyboard, and outputs to output devices like the display.

In high level languages such as Java and Python, we write code using input devices and the output devices give us the high level results such as text, graphics, audio and video. This is in the realm of **software hierarchy** and isn't the focus now. Now we focus on the hardware hierarchy.

All the high level functionalities depend on the low level operations on bits.

<img src="{{ site.baseurl }}/images/cs4ds/io-illustration.png" alt="" align="middle"/>

#### Screen output

There is a part in RAM that's called the **Screen Memory Map** which refreshes many times per second. It directly controls the display on the screen. When we need to display something, we manipulate this part of the memory.

<img src="{{ site.baseurl }}/images/cs4ds/screenmemomap.png" alt="" align="middle"/>

The Hack computer assumes a physical screen of 256x512 pixels black and white. In the memory, we use a chunk of 16-bit registers to represent this pixel matrix.

Since one *word* is 16-bit, and one row in the screen is 512 pixels, we need `512/16=32` words to represent a row. Doing some math we can calculate the memory address of each pixel.

Note that the whole screen memory chunk resides inside the big RAM and it has a starting address. We use a chip `Screen[]` to add the starting base address to the pixel coordinates to get the actual memory address.

<img src="{{ site.baseurl }}/images/cs4ds/scrmap.png" alt="" align="middle"/>

You might be shocked how much work we need to do just to manipulate one pixel. That's the reality at low level!

#### Keyboard input

A physical keyboard is connected to the Keyboard Memory Map in RAM. The good thing is that **we only need one 16-bit register to represent the keyboard**.

<img src="{{ site.baseurl }}/images/cs4ds/keyboardmemmap.png" alt="" align="middle"/>

Each key has a binary **scan code** that goes into the keyboard memory map. If the keyboard is not pressed, the scan code is 0.

To see what key is pressed, just probe the keyboard chip. In the Hack computer, the keyboard memory map is at RAM[24576].

Here is a complete scan code mapping for the Hack computer keyboard:

<img src="{{ site.baseurl }}/images/cs4ds/scancode.png" alt="" align="middle"/>

Notice that the keyboard memory map has 16 bits, it can represent $2^{16}=65536$ different keys which is more than enough even for unicode characters.

#### Draw a rectangle with Hack programming

<img src="{{ site.baseurl }}/images/cs4ds/inputoutputhack.png" alt="" align="middle"/>

Let's consider the "hello world" program of computer graphics: drawing a rectangle.

<img src="{{ site.baseurl }}/images/cs4ds/drawrec.png" alt="" align="middle"/>

Let's focus on the pseudocode. The goal is to manipulate the Screen Memory Map to show the rectangle.

<img src="{{ site.baseurl }}/images/cs4ds/recpseudo.png" alt="" align="middle"/>

The following is the real Hack code

<img src="{{ site.baseurl }}/images/cs4ds/reccode.png" alt="" align="middle"/>

### Compiler: translates high level language to machine code

This is *From NAND to TETRIS part I* and we don't concern ourselves with the compiler which translates a high level language to machine code. In part II, there's material showing how to write a compiler and an operating system.

<img src="{{ site.baseurl }}/images/cs4ds/compiler.png" alt="" align="middle"/>

### Comments

Hack is a simplified version of the machine language. The machine language that controls our day-to-day personal computers is more complex and has more features such as floating point arithmetic. However, we can always use software to expand the capabilities of the machine language, and that is in the part II of *From NAND to TETRIS*.

## Level 5: Computer architecture




## Level 6: Assembler













## Reference

[Coursera From NAND To TETRIS](https://www.coursera.org/learn/build-a-computer)

