"""Correlation of the three-level model against measured data (see CONTRACTS.md).

Modules:
  metrics   error metrics (signed % error, MAPE, max |% error|, Pearson r)
  loaders   readers for data/<gpu_tag>/ files (robust to missing / partial files)
  backend   lazy binding to perfsim.microarch / ops_model / engine, so this package
            imports even when those modules are absent (tests inject stubs)
  correlate the Level 1/2/3 + LOGO + attribution report (CLI)
"""
