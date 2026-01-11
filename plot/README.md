# Plot Generation

To generate the MMLU and all-benchmark average plots along with the final scores table used for the paper:

```bash
python dataset_scaling_mmlu-paper.py & python dataset_scaling-paper.py & wait
```

This will generate:
- `figures/dataset_scaling_mmlu_paper.pdf` - MMLU benchmark scaling plot
- `figures/dataset_scaling_paper.pdf` - All-benchmark average scaling plot
- `figures/final_scores_table.csv` - Final scores table (CSV)
- `figures/final_scores_table.tex` - Final scores table (LaTeX)
