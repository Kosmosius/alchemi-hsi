# EVAL

The main evaluation path assumes a model trained with the alignment trainer.
After running `alchemi align train`, checkpoints contain the encoder towers and
(optional) cycle/band-depth heads. Typical evaluation steps:

1. Load the trained encoder produced by `AlignmentTrainer`.
2. Tokenize new lab↔sensor pairs with `BandTokenizer` using the same config as
   training.
3. Compute retrieval metrics (e.g., `retrieval_at_k`) or spectral-angle deltas
   using utilities in `alchemi.eval`.

The synthetic MAE harness (`alchemi pretrain-mae`) primarily reports throughput
and reconstruction loss on random tokens; it is not intended for downstream
retrieval evaluation.
