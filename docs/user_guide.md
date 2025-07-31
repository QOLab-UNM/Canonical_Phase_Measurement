# User Guide: Scientific Linear Algebra Helpers

## Overview

This package provides efficient Python routines for advanced linear algebra and quantum information tasks, including:
- Computing absolute difference sets
- Identifying non-divisors for sets
- Constructing quantum Fourier transform (QFT) projectors
- Reducing convex combinations via Carathéodory's theorem
- Resolving phase states for the identity operation

These tools are intended for use in scientific research, particularly in quantum information theory, matrix analysis, and related fields.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   cd YOUR-REPO-NAME
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Main Functions and Usage

### 1. `abs_diff_set(K)`

**Purpose:**  
Compute all unique absolute differences |a - b| for a ≠ b in the input array.

**Example:**
```python
from src.linear_algebra_helpers import abs_diff_set
K = [1, 3, 5, 7]
D = abs_diff_set(K)
print(D)  # Output: [2 4 6]
```

---

### 2. `smallest_non_divisor(S)`

**Purpose:**  
Find the smallest integer ≥ 2 that does not divide any element of S.

**Example:**
```python
from src.linear_algebra_helpers import smallest_non_divisor
S = [6, 10, 15]
m = smallest_non_divisor(S)
print(m)  # Output: 7
```

---

### 3. `qft_projectors(K, q)`

**Purpose:**  
Calculate QFT projectors for the basis `K` and dimension `q`.

**Example:**
```python
from src.linear_algebra_helpers import qft_projectors
K = [0, 1, 2]
q = 3
proj = qft_projectors(K, q)
print(proj.shape)  # Output: (9, 3)
```

---

### 4. `find_zero_sum_combination(V)`

**Purpose:**  
Find a nontrivial vector `lambda` such that `V @ lambda = 0` and `sum(lambda) = 0`.

**Example:**
```python
from src.linear_algebra_helpers import find_zero_sum_combination
import numpy as np
V = np.array([[1, 2, 3], [4, 5, 6]])
lam = find_zero_sum_combination(V)
print(lam)
```

---

### 5. `caratheodory_reduce(X, p)`

**Purpose:**  
Reduce a convex combination to at most d+1 points.

**Example:**
```python
from src.linear_algebra_helpers import caratheodory_reduce
import numpy as np
X = np.array([[0, 1, 0], [0, 0, 1]], dtype=float)
p = np.array([0.5, 0.25, 0.25])
Xr, pr, idx = caratheodory_reduce(X, p)
print(Xr)
print(pr)
print(idx)
```

---

### 6. `angles(K)`

**Purpose:**  
Compute angles whose phase states resolve the identity in fractions of 2π.

**Example:**
```python
from src.linear_algebra_helpers import angles
K = [2, 5, 7]
ang, r = angles(K)
print("Angles (fractions of 2*pi):", ang)
print("r:", r)
```

---

## Running the Unit Tests

Unit tests are provided in the `tests/` directory.  
To run all tests:

```bash
python -m unittest discover tests
```

---

## Citing This Repository

If you use this code in your research, please cite it as described in the [README.md](../README.md) and [CITATION.cff](../CITATION.cff).

---

## Support & Contributions

- For issues, questions, or requests, please open an [issue](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME/issues).
- Contributions are welcome! Please see `CONTRIBUTING.md` for details.

---

## License

This package is released under the MIT License. See [LICENSE](../LICENSE) for details.