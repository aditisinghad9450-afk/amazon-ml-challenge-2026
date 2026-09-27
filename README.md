# Amazon ML Challenge 2026 – Business Entity Resolution

Match every Source-1 business record to all records in Source 2 / Source 3 that describe
the same business. Metric: F0.5 (precision-weighted), so the pipeline is tuned to prefer
precision.

## Pipeline

| Step | File | What it does |
|---|---|---|
| Normalise | `src/normalize.py` | lowercases, strips punctuation, removes legal suffixes (LLC, Pvt Ltd…), canonicalises address abbreviations, sorts address tokens (addresses arrive in shuffled order), extracts house numbers |
| Blocking | `src/blocking.py` | hashed blocking keys: rare-name-token pairs, name token + house no., street word + house no., name prefix + house no. Oversized blocks dropped. Prints blocking recall. |
| Pruning | `src/pipeline.py` | cheap fuzzy score on all pairs, keep top-8 per (S1 record, source) |
| Features | `src/features.py` | ~25 RapidFuzz similarities on names/addresses, number overlap, plus context features (rank among the S1's candidates, and among S1s competing for the same target record) |
| Model | `src/pipeline.py` | LightGBM binary classifier, early stopping on held-out S1 records |
| Decision | `src/pipeline.py` | threshold tuned for F0.5 on validation; each S2/S3 record is assigned to at most one S1 record |

## Run

```bash
pip install -r requirements.txt
python src/pipeline.py train     # prints blocking recall + validation F0.5, saves models/
python src/pipeline.py predict   # writes output/submission.tsv
# or: python src/pipeline.py all
```

Data layout: `dataset/train/train_source{1,2,3}.tsv`, `dataset/train/train_ground_truth.tsv`,
`dataset/test/test_source{1,2,3}.tsv` (`.tsv.gz` also works). Normalised sources are cached in
`cache/` as parquet, so reruns skip the slow step.

Output format (`output/submission.tsv`): `source1_entity_id<TAB>matched_entity_ids` with
comma-separated ids, empty when no match — same as the ground-truth file.

## Tuning knobs (top of `src/pipeline.py` / `src/blocking.py`)

- `TOP_K` – candidates kept per S1 per source (recall vs. speed)
- `MAX_BLOCK` – max block size in blocking (raise for recall, lower for speed/memory)
- `TRAIN_S1_SAMPLE` – S1 records used for training; lower it if RAM is tight

Needs ~6–8 GB RAM on the full data. If a machine runs out of memory, lower `TRAIN_S1_SAMPLE`
and `MAX_BLOCK`, or run it on Colab/Kaggle.
