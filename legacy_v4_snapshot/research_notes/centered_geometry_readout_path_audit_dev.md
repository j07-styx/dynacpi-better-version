# Centered geometry readout-path audit (engineering only)

The frozen `SequenceFoundation.predict_tokens` uses **both** an arithmetic mean over valid residues and compound-query multihead attention over residue tokens, followed by normalization and a prediction head.

A controlled unit test demonstrates:

1. `CenteredGeometry` produces a zero-mean residual over structure-covered positions.
2. `CompoundConditionedGeometry` initializes its last gate weights to zero, yielding the same sigmoid gate on all covered residues at initialization.
3. The resulting correction sums to zero, so the **mean-pooling branch cannot detect it** at initialization.
4. The **attention branch can detect it** because its weights vary across residues. Therefore the whole predictor is not mathematically blind to centered geometry.

This mechanism is consistent with the observed near-zero prediction shifts at early epochs but does not prove their cause. The prior seed-1125 diagnostics showed ~2.9% gate activation and very small changes in predictions at epoch 1.

**Design implication:** Before increasing the geometry residual amplitude, instrument attention-specific sensitivity and compare alternative compound-conditioned structural summaries using strict development-only controls. Do not tune repeatedly on the same four validation proteins and call it independent evidence.

Reproduce: `python tests_v4/centered_geometry_readout_sensitivity_test.py`.
