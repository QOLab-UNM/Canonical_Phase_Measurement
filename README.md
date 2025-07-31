# Algorithm to find angles for the construction of optimal phase measurements

A collection of Python functions to find the angles of the convex combination of Theorem 1 for the paper: 

## Features

- Efficient computation of absolute difference sets
- Finding the smallest non-divisor of a set
- Quantum Fourier transform projectors
- Carathéodory reduction for convex combinations
- Angle computation for phase state resolution
- Includes unit tests for each module

## Installation

```bash
git clone https://github.com/QOLab-UNM/Canonical_Phase_Measurement.git
cd YOUR-REPO-NAME
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from src.div_POVM (
    abs_diff_set, smallest_non_divisor, qft_projectors,
    find_zero_sum_combination, caratheodory_reduce, angles
)

K = [2, 5, 7]
ang, r = angles(K)
print("Angles (fractions of 2*pi):", ang)
print("r:", r)
```

See the [`tests/test_div_POVM.py`](tests/test_div_POVM.py) file for more usage examples.

## Function Reference

- **abs_diff_set(K):** Compute unique absolute differences for a list.
- **smallest_non_divisor(S):** Find the smallest integer not dividing any element of S.
- **qft_projectors(K, q):** Generate QFT projectors.
- **find_zero_sum_combination(V):** Find a nontrivial vector in the null space of V with zero sum.
- **caratheodory_reduce(X, p):** Reduce a convex combination to at most d+1 points.
- **angles(K):** Compute angles for phase state resolution.

Full docstrings are provided in the source code.

## Testing

To run the unit tests:

```bash
python -m unittest discover tests
```

## Citation

If you use this code in your research, please cite:

> Your Name et al. (2025). "Title of Your Paper." *Journal/Conference Name*. [arXiv/DOI link]

BibTeX:
```bibtex
@article{Your2025Paper,
  author = {Your Name and Collaborators},
  title = {Title of Your Paper},
  journal = {Journal/Conference Name},
  year = {2025},
  doi = {DOI or arXiv link here}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Questions or feedback? Open an [issue](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME/issues) or email your.email@domain.com.
