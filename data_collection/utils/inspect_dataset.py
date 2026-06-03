import h5py
import numpy as np

def inspect_hdf5(file_path):
    print(f"🔍 Ispezione del file dataset: {file_path}\n" + "="*50)
    
    with h5py.File(file_path, "r") as f:
        # 1. Controlla i metadati globali
        if "metadata" in f:
            print("📜 [METADATI GLOBALI]")
            meta = f["metadata"]
            for key in meta.keys():
                # Gestione decodifica stringhe in HDF5
                val = meta[key][()]
                if isinstance(val, bytes):
                    val = val.decode('utf-8')
                print(f"  ├── {key}: {val}")
        else:
            print("⚠️ Attenzione: Gruppo metadata non trovato!")
            
        print("\n📦 [STRUTTURA CAMPIONI]")
        # Prendi tutte le chiavi escludendo i metadati
        entries = [k for k in f.keys() if k.startswith("entry_")]
        print(f"  ├── Numero totale di entries valide trovate: {len(entries)}")
        
        if len(entries) == 0:
            print("❌ Errore: Nessun record di tipo 'entry_xxxxxx' presente nel file.")
            return
            
        # Ispeziona la prima entry come campione rappresentativo
        sample_key = entries[0]
        print(f"  └── Ispezione campione di test: '{sample_key}'")
        g = f[sample_key]
        
        for dataset_name in g.keys():
            ds = g[dataset_name]
            shape = ds.shape
            dtype = ds.dtype
            
            # Gestione speciale per la lettura di stringhe (come object_id)
            if dtype == object:
                val_preview = ds[()].decode('utf-8') if isinstance(ds[()], bytes) else ds[()]
                print(f"      ├── {dataset_name:<18} -> Tipo: Stringa | Valore: '{val_preview}'")
            else:
                # Anteprima dei valori numerici per controllo sanità
                arr = np.array(ds)
                min_val, max_val = arr.min(), arr.max()
                print(f"      ├── {dataset_name:<18} -> Shape: {str(shape):<15} | Dtype: {str(dtype):<8} | Range: [{min_val}, {max_val}]")

if __name__ == "__main__":
    inspect_hdf5("dataset_chunk_primitive.h5")
