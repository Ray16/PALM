# Additional returned-cut diagnostics

As a post-hoc diagnostic of the native-split extension, audit the fixed rank-256
factor at the saved returned partitions for all three seeds and all datasets.
This diagnostic does not select ranks, methods, datasets or model outcomes.

Reconstruct only sampled factor rows using saved landmark identities, the same
double-precision landmark eigendecomposition and eigenvalue threshold, and the
producer's float32 row storage. A small-instance regression against the actual
producer factor checks this reconstruction. No full factor or n-by-n matrix is
needed. Share the factor reconstruction across exact/corridor cases at equal
seed only after checking their landmark identities agree.

Use the common uniform distinct-entity pair sample and exact original packed-
bit Tanimoto to estimate the signed crossing-cut residual. Report its absolute
value divided by original kernel mass and by the original returned cut, with
approximate block-bootstrap Monte Carlo intervals. If a signed interval covers
zero, its absolute-error lower endpoint is zero. These are returned-partition
diagnostics, not optimization gaps, uniform error certificates or prospective
confirmatory tests. Save source/row identities for each audit.
