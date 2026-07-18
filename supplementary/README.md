# Supplementary experiments

Code for experiments outside the main benchmark table belongs in this
directory, including agent-number, review-round, confidence-calibration,
role-diversity, heterogeneous-model, and case-study analyses.

Main-table experiment implementations remain in the project root.

Paper Table 5 heterogeneous-LLM reproduction:

```bash
./supplementary/run_table5_heterogeneous.sh --time-flag 0719
python supplementary/evaluate_table5_heterogeneous.py --time-flag 0719
```
