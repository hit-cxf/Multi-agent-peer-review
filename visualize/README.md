# Visualization code

All scripts that turn experiment results into figures belong in this directory.
By default, generated figures must be written to `../pics/`.

Every plotting script must print the complete aggregated data used by the
figure while drawing. Use a stable tab-separated table with explicit units,
sample counts, exclusions, and source files where applicable, followed by the
generated output paths.

## Shared color semantics

All figures must import colors from `style.py`. Semantic mappings remain stable:

- neutral / unchanged / ideal reference: `#CBD5E1`
- incorrect / harmful: `#D95F59`
- correct / beneficial / feedback accuracy: `#2A9D6F`
- confidence / unresolved incorrect change: `#E9A23B`

For Figure 5, correct feedback is green, incorrect feedback is red, verbalized
confidence is orange, and the ideal calibration reference is gray.

## Figure scripts

- `plot_answer_transitions.py`: answer-change analysis.
- `plot_confidence_calibration.py`: Figure 5 confidence distribution,
  accuracy-confidence gap, reliability diagrams, ACC, AUROC, and ECE.
