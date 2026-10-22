from pathlib import Path
import numpy as np
DATA=Path(__file__).resolve().parents[2]/"low_rank/data/chembl"
def load(n):
 packed=np.load(DATA/"ecfp2048_packed.npy",mmap_mode="r")
 idx=np.sort(np.random.default_rng(0).permutation(packed.shape[0])[:n])
 return np.unpackbits(np.asarray(packed[idx]),axis=1).astype(np.float32)
