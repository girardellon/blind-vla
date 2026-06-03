import h5py

with h5py.File("test_sample_reale.h5", "r") as f:
    print("--- Contenuto File HDF5 ---")
    for key in f.keys():
        print(f"Gruppo trovato: {key}")
        gruppo = f[key]
        for d_key in gruppo.keys():
            dataset = gruppo[d_key]
            print(f"  -> Dataset: {d_key} | Shape: {dataset.shape} | Tipo: {dataset.dtype}")
