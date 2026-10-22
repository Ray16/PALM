"""One unchanged benchmark row using checksum-verified, precomputed inputs."""
import argparse
import revision_benchmark as benchmark
import cache_revision_input as cache

p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True)
p.add_argument('--method',required=True)
p.add_argument('--epsilon',required=True,type=float)
p.add_argument('--seed',required=True,type=int)
a=p.parse_args()
benchmark.study.prepare=cache.prepare
benchmark.run(a.dataset,[a.method],[a.epsilon],[a.seed])
