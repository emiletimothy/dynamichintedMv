import numpy as np

filename = "factorizations_f2.npz"
with open(filename, 'rb') as f:
    factorizations = dict(np.load(f, allow_pickle=True))

print("Sample tensors:")
for i, (tensor_key, tensor_data) in enumerate(factorizations.items()):
    if i < 5:
        print(f"\nKey: {tensor_key}")
        print(f"  Type: {type(tensor_data)}")
        print(f"  Shape: {getattr(tensor_data, 'shape', 'N/A')}")
        print(f"  Dtype: {getattr(tensor_data, 'dtype', 'N/A')}")
        if hasattr(tensor_data, 'shape') and len(tensor_data.shape) == 1:
            print(f"  Contents (first few): {tensor_data[:min(3, len(tensor_data))]}")
            if len(tensor_data) > 0:
                print(f"  First element type: {type(tensor_data[0])}")
                if hasattr(tensor_data[0], 'shape'):
                    print(f"  First element shape: {tensor_data[0].shape}")
