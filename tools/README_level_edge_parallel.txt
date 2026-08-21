Level edge audit parallel execution

Use tools/run_level_edge_audit_16w.sh.
Default WORKERS=16. Each pair runs in its own Python process and writes an isolated part directory. The parent process aggregates all pair results into the final summary after every worker finishes.
