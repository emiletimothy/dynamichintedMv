from typing import List, Union
import numpy as np

class TensorSubproductAnalyzer:
    def __init__(self, tensor_data, T1=None, T2=None, T3=None) -> None:
        self.tensor = tensor_data        
        if len(tensor_data.shape) == 3:
            self.format = '3d'
            self.size = tensor_data.shape[2]
            if T1 is None or T3 is None:
                self.output_dim = int(np.sqrt(tensor_data.shape[1]))
                self.T1 = self.output_dim
                self.T3 = self.output_dim
            else:
                self.T1 = T1
                self.T3 = T3
        else:
            self.format = 'object'
            self.size = tensor_data[0].shape[1]  # Number of columns in first matrix
            if T1 is None or T3 is None:
                raise ValueError("T1, T2, T3 must be provided for object array format")
            self.T1 = T1
            self.T3 = T3
            
        self.T2 = T2 if T2 is not None else self.T1
        self.dimension = self.T3
        self.Gs = [self.count_output_support()]

    def count_output_support(self) -> List[int]:
        if self.format == '3d':
            output_slice = self.tensor[2] 
            num_rows = self.T1
            num_cols = self.T3 
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
        else:
            output_matrix = self.tensor[2]  
            num_rows = self.T1
            num_cols = self.T3
            num_subproducts = output_matrix.shape[1]
            support_sizes = []
            for subproduct_idx in range(num_subproducts):
                columns_affected = 0
                for col_idx in range(num_cols):
                    column_entries = list(range(col_idx, num_rows * num_cols, num_cols))
                    affects_this_column = any(output_matrix[entry_idx, subproduct_idx] != 0 
                                              for entry_idx in column_entries)
                    if affects_this_column:
                        columns_affected += 1
                support_sizes.append(columns_affected)
        
        Gs = [0] * num_cols
        for size in support_sizes:
            if size > 0 and size <= num_cols:
                Gs[size - 1] += 1
        return Gs
    