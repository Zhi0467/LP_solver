# LP_solver

A Linear Programming solver implementing the Simplex method with Big-M initialization.

## Requirements
- Python 3.8+
- NumPy
- Matplotlib (for runtime analysis)

## Features 
- Solves Linear Programming problems using the Simplex method
- Handles inequality and equality constraints
- Automatically converts problems to standard form
- Includes feasibility checking
- Runtime scaling analysis tools
- Solution verification capabilities

## Usage
store your input file as "LP_input.in" in the data folder, and run main.py to solve the LP problem provided in that file.


### Input File Format
n m # number of variables, number of constraints  
c1 c2 ... cn # objective coefficients  
a11 a12 ... a1n b1 <= # constraint 1 (<=, >= or =)  
a21 a22 ... a2n b2 <= # constraint 2  
...  
am1 am2 ... amn bm <= # constraint m  
