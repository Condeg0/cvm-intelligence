Run the extraction validation pipeline and produce a summary report.

Steps:
1. Check that `data/cvm_metrics.db` exists and has data in the `metrics` table
2. Query all metrics and compute:
   - **Exact Match Rate:** % where `match_status = 'exact'`
   - **MAPE:** Mean Absolute Percentage Error across all `percentage_error` values where status is 'close'
   - **Coverage:** % of expected metrics that were extracted (not 'missing')
3. Break down failures by category (`match_status` distribution)
4. Show the 10 worst extraction errors (highest `percentage_error`)
5. Show per-company extraction success rates
6. Show per-metric extraction success rates (which metrics are hardest to extract?)

Print results as a formatted table. If targets are met (>90% exact match, <2% MAPE, >85% coverage), say so explicitly. If not, identify the biggest failure categories to fix.

If the database doesn't exist or is empty, say so and indicate this is a Phase 3 deliverable.
