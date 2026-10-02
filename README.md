# No-Code ML

Point it at a table or text dataset and get a model you own, trained on your own machine. Before it shows any score, it tells you when that score would be lying.

```
nocodeml train customers.csv --target churned
```

The output of each run:

1. **Validity checks**, run before any score. Columns that leak the target, ID columns and date columns are excluded with a reason. You also get warnings for class imbalance, too little data, conflicting duplicate rows and rows with no target.
2. **Three model tiers**, compared under the same cross-validation:
   - **Tier 0 (floor):** always predicts the most common or mean value. Every other score is judged against it.
   - **Tier 1 (zero training):** a tabular foundation model ([TabICL](https://github.com/soda-inria/tabicl)) that predicts with no training.
   - **Tier 2 (cheap training):** gradient boosting for tables, or TF-IDF plus a linear model when the data has text columns.

   A trained model only wins if it beats the zero-training model by more than its own cross-validation noise.
3. **A run folder** at `runs/<timestamp>-<id>/` containing `report.md`, `report.json` and `model.joblib`. Each run gets its own folder, so several runs can happen at the same time.

## Install

```
pip install -e ".[fm]"   # [fm] adds the zero-training tier (pulls in CPU torch)
pip install -e .         # lighter: tiers 0 and 2 only
```

## Use

```
nocodeml train data.csv --target COLUMN                 # CSV, XLSX or Parquet
nocodeml train data.csv --target COLUMN --keep COL      # keep a column a check excluded
nocodeml predict runs/<run-id> new_rows.csv --out preds.csv
```

The zero-training tier is slow on CPU. Set `NOCODEML_SKIP_FM=1` to skip it.

`model.joblib` is a pickle. Only load runs you created yourself.

## Develop

```
pip install -e ".[dev,fm]"
python -m pytest -q
```

The roadmap lives in `.claude/prds/no-code-ml-v2.prd.md`. Next up is "replace my LLM prompt with a small model" (milestone 2).

## License

MIT

## Contact

[vishaltyagi.9468@gmail.com](mailto:vishaltyagi.9468@gmail.com)
