# Tensor Subproduct Analyzer

Analysis tool for matrix multiplication tensors, calculating entanglement measures and CP-rank decompositions.

## Overview

This project analyzes tensor decompositions for matrix multiplication, specifically:
- Computing G_i histograms (number of subproducts affecting exactly i columns)
- Calculating CP-rank (number of subproducts in the decomposition)
- Computing entanglement measures

## Files

- `tensor_subproduct_analyzer.py`: Core analyzer class for processing tensor decompositions
- `compute_entanglement_rank.py`: Main script to process tensors and compute metrics
- `factorizations_f2.npz`: Tensor decomposition data
- `tensor_analysis_results.csv`: Output results

## Usage

```bash
python3 compute_entanglement_rank.py
```

This processes any matrix multiplication tensor from the factorizations file and outputs:
- Matrix dimension (T)
- CP-rank
- G_i histogram (distribution of subproduct column support)
- Entanglement measure

Results are saved to `tensor_analysis_results.csv`.

## Requirements

See `requirements.txt` for dependencies.
