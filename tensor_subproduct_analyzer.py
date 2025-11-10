from typing import List, Union
import numpy as np

class TensorSubproductAnalyzer:
    def __init__(self, tensor_data) -> None:
        self.tensor = tensor_data
        self.size = tensor_data.shape[2] 
        self.output_dim = int(np.sqrt(tensor_data.shape[1])) 
        self.dimension = self.output_dim 
        self.Gs = [self.count_output_support()]

    def count_output_support(self) -> List[int]:
        output_slice = self.tensor[2] 
        num_rows = self.output_dim
        num_cols = self.output_dim 
        num_subproducts = output_slice.shape[1]
        support_sizes = []
        for subproduct_idx in range(num_subproducts):
            columns_affected = 0
            for col_idx in range(num_cols):
                column_entries = list(range(col_idx, num_rows * num_cols, num_cols))
                affects_this_column = any(output_slice[entry_idx, subproduct_idx] != 0 
                                          for entry_idx in column_entries)
                if affects_this_column:
                    columns_affected += 1
            support_sizes.append(columns_affected)
        Gs = [0] * num_cols
        for size in support_sizes:
            if size > 0 and size <= num_cols:
                Gs[size - 1] += 1
        return Gs
    
    @staticmethod
    def is_nonzero(value) -> bool:
        if isinstance(value, (int, float)):
            return value != 0
        return True
    
    def analyze_multiple_tensors(self, tensor_list: List) -> List[List[int]]:
        num_columns = len(tensor_list[0][0][0]) 
        column_counts = [0] * num_columns
        for subproduct in tensor_list:
            for col_idx in range(num_columns):
                affects_column = False
                for matrix in subproduct:
                    for row in matrix:
                        if col_idx < len(row) and self.is_nonzero(row[col_idx]):
                            affects_column = True
                            break
                    if affects_column: break
                if affects_column: column_counts[col_idx] += 1
        return column_counts
    
    def analyze_multiple_tensors(self, tensor_list: List) -> List[List[int]]:
        results = []
        for tensor in tensor_list:
            result = self.count_subproducts_per_column(tensor)
            results.append(result)
        return results
    