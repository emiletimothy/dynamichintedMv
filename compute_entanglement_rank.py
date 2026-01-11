from math import e
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import re
import csv
from tensor_subproduct_analyzer import TensorSubproductAnalyzer

def entanglement(Gs, T, rank, T1, T2, T3):
    mu_p = np.sum([Gs[i] * np.log(i+1)/rank for i in range(T)])
    return mu_p, rank, np.log(T1*T2*T3/rank) - mu_p

def entanglement_measure(Gs, T, rank, T1, T2, T3):
    _, _, entanglement_measure = entanglement(Gs, T, rank, T1, T2, T3)
    return entanglement_measure

def tensor_rank(Gs, T):
    return np.sum(Gs)
 
############ Load factorizations ############
filename = "factorizations_r.npz"
with open(filename, 'rb') as f:
  factorizations = dict(np.load(f, allow_pickle=True))

results = []
for tensor_key, tensor_data in factorizations.items():
    # Parse tensor dimensions
    dims = tensor_key.split(',')
    if len(dims) != 3:
        continue
    T1, T2, T3 = int(dims[0]), int(dims[1]), int(dims[2])
    analyzer = TensorSubproductAnalyzer(tensor_data, T1, T2, T3)
    T = analyzer.dimension
    Gs = analyzer.Gs[0]
    cp_rank = analyzer.size
    sum_Gs = tensor_rank(Gs, T)
    ent_measure = entanglement_measure(Gs, T, cp_rank, T1, T2, T3)
    results.append({
        'Tensor': tensor_key,
        'Dimensions': f'{T1}×{T2}×{T3}',
        'Rank': cp_rank,
        'Gs': str(Gs),
        'Entanglement': ent_measure
    })
    if ent_measure > 0:
        print(f"Tensor {tensor_key} has positive entanglement rank {ent_measure}")

# Save to CSV
csv_filename = "tensor_analysis_results.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ['Tensor', 'Dimensions', 'Rank', 'Gs', 'Entanglement']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Results saved to {csv_filename}")