# Preempted wall-clock comparison

The original 72 clock cases check deadlines cooperatively between operations.
Their overrun is retained explicitly, but this is weaker than the requested
equal-wall-clock comparison. Preserve those observations as a sensitivity
protocol and add the same 72 configurations with an externally terminated
timed subprocess. Construction and optimization are both inside its clock.

Each subprocess starts only after imports, loading and a common warm-up.
A feasible initial checkpoint exists before its start signal. An improved
checkpoint is eligible only when its objective evaluation and label-array
serialization finish before the deadline. The supervising process terminates
the optimizer at the deadline, then independently scores the last eligible
checkpoint. Scheduling/termination overshoot is reported, and work completing
after the selection cutoff cannot improve its result. The 1- and 5-second
budgets, datasets, ranks, input-order restart seeds and capacity rules remain
unchanged. Both representations invoke the same one-step current routines.

These are an equal-time resource comparison, distinct from the fixed-iteration
control. Primary package and old producer sources remain unchanged. Pin the
new adapter separately; retain all cooperative results and their actual scope.
