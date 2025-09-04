import numpy as np

# Path to your dataset
npz_path = "jobs/enformer_style_chr22.npz"

# Load .npz file
data = np.load(npz_path, allow_pickle=True)

# List all array keys
print("Contents of the .npz file:")
print("--------------------------")
for key in data.files:
    print(f"{key}: shape = {data[key].shape}, dtype = {data[key].dtype}")

print("\nPreview of first entries:")
print("--------------------------")
for key in data.files:
    print(f"\n[{key}] sample values:")
    arr = data[key]
    if arr.ndim == 1:
        print(arr[:10])  # print first 10 values
    elif arr.ndim == 2:
        print(arr[0][:10])  # print first 10 of first row
    elif arr.ndim == 3:
        print(arr[0, 0, :10])  # example for 3D array
    else:
        print("Array too complex for preview.")
