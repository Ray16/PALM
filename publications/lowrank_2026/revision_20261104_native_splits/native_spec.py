"""Validated four-role specification adapter for the unchanged k-way solver.

The frozen package's ordinary SplitSpec permits two/three roles. Numerical
factor, assignment and refinement functions already accept arbitrary k. This
study adapter preserves PCQM4Mv2's four named roles without editing that package.
"""
import math
from lowrank import SplitSpec


class NativeSplitSpec(SplitSpec):
    def __post_init__(self):
        if len(self.splits)!=4:
            super().__post_init__()
            return
        self.splits=list(self.splits);self.names=list(self.names)
        if len(self.names)!=4 or len(set(self.names))!=4 or any(not isinstance(n,str) or not n for n in self.names):
            raise ValueError('four distinct nonempty partition names are required')
        if any(isinstance(s,bool) or not math.isfinite(s) or s<=0 for s in self.splits):
            raise ValueError('partition weights must be finite and positive')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('nonnegative integer seed required')
        if not math.isfinite(self.epsilon) or not 0<=self.epsilon<1:
            raise ValueError('epsilon must be finite and in [0,1)')
