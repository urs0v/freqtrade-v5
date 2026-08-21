LEVEL EDGE HIGH-ROI V1 FAST PATH

The fast runner preserves the existing event generation, filters, cost model, leverage, portfolio rules, TRAIN/VALID/HIST_TEST splits, and winner selection semantics.
It only parallelizes the expensive post-scan portfolio candidate evaluation and reuses already-written causal_events/stage1/stage2 CSVs.
The production FrozenFakeout websocket dry-run path is not modified.
