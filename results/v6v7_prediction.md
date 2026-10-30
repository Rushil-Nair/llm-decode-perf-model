# v6 / v7 (bank-conflict-free x in shared memory): prediction made before they ran

Written 2026-10-30T00:24:57Z, after the first Lium H100 run (data/h100_lium) and before v6/v7 ran on any GPU.

What the first run showed (results/ncu_deep_h100_lium.md, results/v5_score_h100_sxm.md):
- v5 halved executed instructions per group (7.23 -> 3.62 warp instructions) and issue-slot use (64% -> 32%),
  but gate_proj only went from 1526 to 1604 GB/s (1.05x; the instruction-count prediction was 1.35x).
- v3/v4/v5 all read x from shared memory with a 4-way bank conflict: 3.2M shared wavefronts vs 0.84M ideal
  (3.8x), shared-memory data pipe ~60% busy, and "short scoreboard" (waiting on shared memory) is the top stall.

Prediction:
- The SM-side limit is the conflicted shared-memory reads, not instruction count. Removing the conflicts (v6, v7)
  should move the best gate_proj config up to the model's loaded-latency memory ceiling on this H100,
  ~2.09 TB/s (66% of the 3.15 TB/s stream peak); predicted v7 gate_proj best: 1.9-2.1 TB/s.
- v6 keeps v4's instruction count (plus ~15% swizzle address math), so it may become issue-limited before the
  memory ceiling: predicted between v4 (1.53 TB/s) and v7.
- Shared wavefronts for v6/v7 should drop to about the ideal count (~0.84M on gate_proj) with ~0 bank conflicts.
