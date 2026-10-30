# v7 on the RTX 5090: prediction made before it ran there

Written 2026-10-30T01:27:11Z. Measured so far on the RTX 5090 (data/rtx5090, Vast, Oct 29): v4 best
1538 GB/s on gate_proj = 90% of the 1700 GB/s stream peak; the three-ceiling model puts 77 of 146 configs on the
memory ceiling. On the H100, v7 (no bank conflicts + fewer instructions) gave 1.34x over v4 (2048 vs 1528 GB/s).

Prediction: on the RTX 5090 the same changes give little, because v4 is already close to the memory roof.
- v7 / v4 on gate_proj best config: 0.95x to 1.10x. Hard upper bound from the stream peak: 1700 / 1538 = 1.11x.
- v5 and v6 alone: within +-10% of v4.
- If v7 gains more than 1.11x over v4, the stream probe under-measures the 5090's peak bandwidth.
