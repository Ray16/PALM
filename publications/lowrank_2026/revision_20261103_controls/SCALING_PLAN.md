# Bounded dense-versus-factor scaling control

The molecular pilot establishes that dense optimization can be faster on small
datasets. To measure the size-dependent cost directly, add nested samples of
4,000, 16,000 and 32,000 ChEMBL entities, ranks 64/256/512 and a direct dense
arm, three optimization seeds, ordinary Tanimoto and a 1% capacity corridor.
This 36-case grid uses the same initial assignments, current assignment and
refinement routines and limits as the iteration-controlled molecular study.
The samples follow the fixed seed-zero nested ordering used by the existing
ChEMBL scaling study. The complete packed-input checksum must agree with the
existing study, and selected indices and feature checksums are retained.

Two independent two-core CPU workers, disjoint from the molecular grid, run in
fresh processes. Representation construction is inside the clock; loading,
warm-up and independent full-kernel scoring are outside it. Process RSS is
captured before scoring. All pairs are evaluated for quality; no full matrix
is constructed in factor optimization. This bounded control is distinct from
the existing two-million-entity public-API measurements and must not be used
to reinterpret their hardware or restart settings.
