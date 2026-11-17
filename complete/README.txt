Anonymous research code and records

The original foundation/ and application/ records are retained. The rebuttal/ directory contains the three new experiments and verification. The anonymous review mirror exposes this complete package.

Run in a Python environment with requirements.txt installed:
    python reproduce.py --output fresh_replay

This copies only the new executable sources into a new directory, runs the three full grids, and runs verification. It preserves all supplied results and protocols. The output directory must not already exist. The CPU run uses no paid services and no GPU. WDBC, Digits and Wine are bundled scikit-learn datasets.

New records:
- matched_categorical.json: all 4,800 matched trials, 48 configuration summaries, bound components, counts, active terms, known-vector comparisons.
- natural_sensitivity.json and natural_sensitivity_records.json: all 81 analyses and case-level scores/split indices. The original 40/30/30, cap-two settings replay exactly at the original confidence level.
- continuous_witness.json and continuous_records.json: computable normalizer/KL budget, all 120 audits, saved predictions and witnesses.
- verification.json: disjoint splits, exact score replay, matched component checks, and separate post-execution quadrature verification of the continuous-posterior bound.

Each protocol records its initial design and source hash. The first matched run completed computation but failed when serializing numpy scalar counts; scalar serialization was fixed and the identical grid/seeds rerun, as recorded. Natural-label execution emitted platform matrix-product floating-point warnings; all scores, bounds and original-setting replays were checked and are finite, with the original-setting lower bounds reproduced exactly. No design was changed to select favorable results.

Scope: categorical decision comparisons are controlled experiments with an exact candidate to isolate the witness. Continuous-posterior certificates concern the specified small Bayesian model, not natural patient outcomes. Natural-label sensitivity concerns a training-defined binary context and assumes iid population sampling. Shared fits/splits are not independent model replications. Numerical checks are not independent human verification or full outward floating-point certification.
