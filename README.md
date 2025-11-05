# Optimization Techniques Implementation



## Repository Structure



## Techniques Overview

### 1. Derivative-free Methods
- **Dichotomous Search**: Implementation for single-variable unimodal functions
- **Fibonacci Search**: Optimization using Fibonacci sequence properties
- **Golden-section Search**: Implementation using golden ratio for optimal interval reduction

### 2. Line Search Methods
- **Exact Line Search**: Finds exact minimizer along search direction
- **Inexact Line Search**: Practical implementation with Wolfe conditions

### 3. Numerical Methods
- **Newton's Method**: Uses gradient and Hessian information
- **BFGS Method**: Quasi-Newton method with Hessian approximation

### 4. Penalty and Barrier Methods
- **Barrier Method**: Interior point approach using logarithmic barriers
- **Exact Penalty Method**: L1 penalty implementation for constraints

### 5. Interior Point Methods
- **Dual Interior Point Method**: Implementation focusing on dual form
- **Path-Following PD-IPM**: Primal-dual implementation with path following

### 6. Integer Programming
- Implementation for problems with integer constraints



### Data Input Format



## Requirements
- Python 3.8+
- NumPy
- Pandas
- SciPy
- Matplotlib
- Jupyter Notebook

## Installation
```bash
pip install -r requirements.txt
```

## Requirements & Running Tests

1. Create and activate a Python 3.8+ environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Quick smoke test (verifies required packages can be imported):

```bash
python -c "import numpy,pandas,matplotlib,scipy; print('imports OK')"
```

## References
- [1397] Dichotomous and Fibonacci Search Methods
- [1442, 1443] Line Search Methods
- [1459, 1462] Newton's Method
- [1479, 1480] BFGS Method
- [1553, 1554] Barrier Methods
- [1592, 1593] Exact Penalty Methods
- [1623, 1624] Dual Interior Point Methods
- [1645] Path-Following Methods
- [1522] Integer Programming

## Project Summary / Tracking

An online summary and tracker for this project is available here:

https://docs.google.com/spreadsheets/d/1KoRvEzZNBKlGrbbz5BnxFKAkKGprH4693D0e-W9BJu0/edit?gid=0#gid=0
https://docs.google.com/document/d/1WVbbh_lB0llAEiIojDIiz_YGTuOUylBuHfpV85DIp1Q/edit?tab=t.jr5smitzhxxe

## Problem Statement

The formal problem statement PDF is included at:

`docs/problem_statement.pdf`

Open that PDF to read the full problem description and objectives.

## Notes
