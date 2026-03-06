"""DNA/RNA gene sequence featurization (3 feature sets).

Feature sets:
  - nucleotide_composition: GC content, nucleotide frequencies, GC/AT skew,
                            CpG O/E ratio, melting temperature estimate (no deps)
  - kmer_frequencies: dinucleotide + trinucleotide relative frequencies (no deps)
  - nt_embedding: Nucleotide Transformer or DNABERT-2 embeddings (requires
                  torch + transformers)
  - precomputed_embedding: load embeddings from CSV / .npy / .npz / .pt file
"""

import numpy as np
import pandas as pd
from itertools import product


# ── Nucleotide composition ────────────────────────────────────────────────

def nucleotide_composition(sequence):
    """Global composition features from a nucleotide sequence (DNA or RNA).

    Computed features (11 total):
        length          - number of nucleotides
        freq_A/T/G/C    - mononucleotide relative frequencies (RNA U → T)
        gc_content      - (G + C) / length
        at_content      - (A + T) / length
        gc_skew         - (G - C) / (G + C), strand asymmetry measure
        at_skew         - (A - T) / (A + T), strand asymmetry measure
        cpg_oe          - CpG observed / expected ratio (methylation / regulatory signal)
        melting_temp    - estimated Tm in °C (Wallace rule for <14 bp, else empirical)
    """
    seq = sequence.upper().replace("U", "T")
    n = len(seq)

    zero = {
        "length": 0, "freq_A": 0.0, "freq_T": 0.0, "freq_G": 0.0, "freq_C": 0.0,
        "gc_content": 0.0, "at_content": 0.0, "gc_skew": 0.0, "at_skew": 0.0,
        "cpg_oe": 0.0, "melting_temp": 0.0,
    }
    if n == 0:
        return zero

    nA = seq.count("A")
    nT = seq.count("T")
    nG = seq.count("G")
    nC = seq.count("C")

    gc_denom = nG + nC
    at_denom = nA + nT

    gc_skew = (nG - nC) / gc_denom if gc_denom > 0 else 0.0
    at_skew = (nA - nT) / at_denom if at_denom > 0 else 0.0

    # CpG observed/expected: (count_CpG * n) / (nC * nG)
    cpg_count = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    cpg_expected = (nC * nG) / n
    cpg_oe = cpg_count / cpg_expected if cpg_expected > 0 else 0.0

    # Melting temperature
    if n < 14:
        tm = 4.0 * gc_denom + 2.0 * at_denom          # Wallace rule
    else:
        tm = 64.9 + 41.0 * (gc_denom - 16.4) / n      # empirical formula

    return {
        "length": n,
        "freq_A": nA / n,
        "freq_T": nT / n,
        "freq_G": nG / n,
        "freq_C": nC / n,
        "gc_content": gc_denom / n,
        "at_content": at_denom / n,
        "gc_skew": gc_skew,
        "at_skew": at_skew,
        "cpg_oe": cpg_oe,
        "melting_temp": tm,
    }


# ── K-mer frequencies ─────────────────────────────────────────────────────

def _all_kmers(k, alphabet="ATGC"):
    return ["".join(p) for p in product(alphabet, repeat=k)]


_KMERS_2 = _all_kmers(2)   # 16 dinucleotides
_KMERS_3 = _all_kmers(3)   # 64 trinucleotides (codons)


def kmer_frequencies(sequence):
    """Dinucleotide and trinucleotide relative frequencies (80 features, no deps).

    Sliding-window counts normalised by the number of possible windows.
    These capture local composition bias and are the standard input for
    sequence-based clustering and similarity estimation in genomics.
    """
    seq = sequence.upper().replace("U", "T")
    n = len(seq)
    feats = {}

    # Dinucleotides (k=2): 16 features
    di_counts = {km: 0 for km in _KMERS_2}
    n_di = max(n - 1, 1)
    for i in range(n - 1):
        km = seq[i:i + 2]
        if km in di_counts:
            di_counts[km] += 1
    for km in _KMERS_2:
        feats[f"di_{km}"] = di_counts[km] / n_di

    # Trinucleotides (k=3): 64 features
    tri_counts = {km: 0 for km in _KMERS_3}
    n_tri = max(n - 2, 1)
    for i in range(n - 2):
        km = seq[i:i + 3]
        if km in tri_counts:
            tri_counts[km] += 1
    for km in _KMERS_3:
        feats[f"tri_{km}"] = tri_counts[km] / n_tri

    return feats


# ── DNA language model embeddings ─────────────────────────────────────────

# Model registry: name -> (HuggingFace hub path, embedding dim)
NT_MODELS = {
    "nt_500m_human_ref":      ("InstaDeepAI/nucleotide-transformer-500m-human-ref", 1024),
    "nt_500m_1000g":          ("InstaDeepAI/nucleotide-transformer-500m-1000g", 1024),
    "nt_2500m_multi_species":  ("InstaDeepAI/nucleotide-transformer-2.5b-multi-species", 2560),
    "nt_2500m_1000g":         ("InstaDeepAI/nucleotide-transformer-2.5b-1000g", 2560),
    "dnabert2":               ("zhihan1996/DNABERT-2-117M", 768),
}


class NucleotideTransformerEmbedder:
    """Lazy-loaded DNA language model for generating sequence embeddings.

    Mean-pools last hidden state over sequence positions to produce a
    fixed-size embedding per gene sequence. Supports the Nucleotide
    Transformer family (InstaDeep) and DNABERT-2.
    """

    def __init__(self, model_name="nt_500m_human_ref", batch_size=8, max_length=512):
        if model_name not in NT_MODELS:
            raise ValueError(
                f"Unknown DNA model '{model_name}'. "
                f"Available: {list(NT_MODELS.keys())}"
            )
        self.hub_name, self.embed_dim = NT_MODELS[model_name]
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length   # in tokens (NT uses 6-nt tokens)
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"  Loading DNA model: {self.hub_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hub_name, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.hub_name, trust_remote_code=True
        )

        if torch.cuda.is_available():
            self._model = self._model.cuda()
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("  Using CPU (no GPU detected)")

        self._model.eval()

    def embed_sequences(self, sequences):
        """Mean-pool embeddings for a list of nucleotide sequences.

        Args:
            sequences: list of DNA/RNA strings

        Returns:
            np.ndarray of shape (len(sequences), embed_dim)
        """
        import torch

        self._load_model()
        device = next(self._model.parameters()).device

        all_embeddings = []
        for i in range(0, len(sequences), self.batch_size):
            batch = sequences[i:i + self.batch_size]
            # NT tokenizes in 6-nt chunks; truncate raw sequence accordingly
            batch = [s[:self.max_length * 6] for s in batch]

            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length + 2,   # +2 for BOS/EOS
            ).to(device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            hidden = outputs.last_hidden_state   # (B, L, D)
            mask = inputs["attention_mask"]       # (B, L)

            mask_exp = mask.unsqueeze(-1).float()
            summed = (hidden * mask_exp).sum(dim=1)
            counts = mask_exp.sum(dim=1).clamp(min=1)
            embeddings = (summed / counts).cpu().numpy()
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0)


# Module-level embedder instance (lazy, shared across calls)
_nt_embedder = None


def _get_nt_embedder(model_name="nt_500m_human_ref", batch_size=8):
    global _nt_embedder
    if _nt_embedder is None or _nt_embedder.model_name != model_name:
        _nt_embedder = NucleotideTransformerEmbedder(
            model_name=model_name, batch_size=batch_size
        )
    return _nt_embedder


def nt_embedding(sequence):
    """Sentinel — batched NT embeddings are computed by compute_gene_features."""
    raise RuntimeError(
        "nt_embedding should not be called directly. "
        "It is computed in batch by compute_gene_features."
    )


def precomputed_embedding(sequence):
    """Sentinel — pre-computed embeddings are loaded by compute_gene_features."""
    raise RuntimeError(
        "precomputed_embedding should not be called directly. "
        "It is loaded from file by compute_gene_features."
    )


# Registry of gene feature sets
GENE_FEATURE_SETS = {
    "nucleotide_composition": nucleotide_composition,
    "kmer_frequencies": kmer_frequencies,
    "nt_embedding": nt_embedding,
    "precomputed_embedding": precomputed_embedding,
}


# ── Main entry point ──────────────────────────────────────────────────────

def compute_gene_features(
    entities,
    feature_sets=None,
    nt_model="nt_500m_human_ref",
    nt_batch_size=8,
    embedding_file=None,
):
    """Compute gene/nucleotide features for a dict of {entity_id: sequence}.

    Args:
        entities: dict mapping entity ID to nucleotide sequence (DNA or RNA).
                  RNA sequences (containing U) are automatically normalised to DNA.
        feature_sets: list of feature set names, or None to use defaults
                      ["nucleotide_composition", "kmer_frequencies"].
                      Options:
                        "nucleotide_composition" - GC%, skew, CpG O/E, Tm (11 features)
                        "kmer_frequencies"       - di + trinucleotide freqs (80 features)
                        "nt_embedding"           - Nucleotide Transformer embeddings
                                                   (1024 or 2560 dims, requires torch)
                        "precomputed_embedding"  - load from embedding_file
        nt_model: DNA language model for "nt_embedding" feature set.
                  Available: nt_500m_human_ref (default), nt_500m_1000g,
                             nt_2500m_multi_species, nt_2500m_1000g, dnabert2
        nt_batch_size: batch size for model inference
        embedding_file: path to pre-computed embeddings.
                        Supported: .csv (entity_id index), .npy, .npz (ids + embeddings),
                                   .pt / .pth (dict {id: tensor} or bare tensor)

    Returns:
        DataFrame with entity_id as index, feature columns as floats
    """
    if feature_sets is None:
        feature_sets = ["nucleotide_composition", "kmer_frequencies"]

    entity_ids = list(entities.keys())
    simple_sets = [fs for fs in feature_sets
                   if fs not in ("nt_embedding", "precomputed_embedding")]

    # Per-sequence features
    rows = {}
    for entity_id in entity_ids:
        sequence = entities[entity_id]
        feats = {}
        for fs_name in simple_sets:
            feats.update(GENE_FEATURE_SETS[fs_name](sequence))
        rows[entity_id] = feats

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "entity_id"
    df = df.fillna(0)

    # Nucleotide Transformer / DNABERT-2 embedding (batch)
    if "nt_embedding" in feature_sets:
        embedder = _get_nt_embedder(model_name=nt_model, batch_size=nt_batch_size)
        sequences = [entities[eid] for eid in entity_ids]
        print(f"  Computing NT embeddings ({nt_model}) for {len(sequences)} sequences...")
        emb_matrix = embedder.embed_sequences(sequences)
        emb_cols = [f"nt_{i}" for i in range(emb_matrix.shape[1])]
        emb_df = pd.DataFrame(emb_matrix, index=entity_ids, columns=emb_cols)
        emb_df.index.name = "entity_id"
        df = pd.concat([df, emb_df], axis=1)
        print(f"  NT embeddings: {emb_matrix.shape[1]} dimensions")

    # Pre-computed embeddings from file
    if "precomputed_embedding" in feature_sets:
        if embedding_file is None:
            raise ValueError(
                "embedding_file must be specified when using 'precomputed_embedding'"
            )
        print(f"  Loading pre-computed embeddings from {embedding_file}")

        if embedding_file.endswith(".npz"):
            data = np.load(embedding_file)
            emb_ids = list(data["ids"])
            emb_matrix = data["embeddings"]
        elif embedding_file.endswith(".npy"):
            emb_matrix = np.load(embedding_file)
            emb_ids = entity_ids
        elif embedding_file.endswith(".csv"):
            emb_raw = pd.read_csv(embedding_file, index_col=0)
            emb_ids = list(emb_raw.index.astype(str))
            emb_matrix = emb_raw.values
        elif embedding_file.endswith((".pt", ".pth")):
            import torch
            data = torch.load(embedding_file, map_location="cpu", weights_only=False)
            if isinstance(data, dict):
                emb_ids = list(data.keys())
                emb_matrix = np.stack([
                    v.numpy() if hasattr(v, "numpy") else np.array(v)
                    for v in data.values()
                ])
            else:
                emb_matrix = data.numpy() if hasattr(data, "numpy") else np.array(data)
                emb_ids = entity_ids
        else:
            raise ValueError(f"Unsupported embedding file format: {embedding_file}")

        id_to_idx = {str(eid): i for i, eid in enumerate(emb_ids)}
        aligned_embs = []
        for eid in entity_ids:
            if str(eid) in id_to_idx:
                aligned_embs.append(emb_matrix[id_to_idx[str(eid)]])
            else:
                print(f"    WARNING: No embedding found for entity '{eid}', using zeros")
                aligned_embs.append(np.zeros(emb_matrix.shape[1]))

        emb_aligned = np.stack(aligned_embs)
        emb_cols = [f"emb_{i}" for i in range(emb_aligned.shape[1])]
        emb_df = pd.DataFrame(emb_aligned, index=entity_ids, columns=emb_cols)
        emb_df.index.name = "entity_id"
        df = pd.concat([df, emb_df], axis=1)
        print(f"  Pre-computed embeddings: {emb_aligned.shape[1]} dimensions")

    return df
