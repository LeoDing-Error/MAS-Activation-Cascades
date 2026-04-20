# Contrastive Pairs

Generated contrastive-pair JSON files live here.

Recommended workflow:

```bash
python3 scripts/build_ta2_pairs.py \
  --dataset harmful \
  --output data/contrastive_pairs/ta2_harmful_pairs.json
```

This uses the full TA2 harmful dataset by default. Pass `--limit <N>` only for smoke-test subsets.

The generated JSON files are ignored by git because they may contain unsafe evaluation prompts drawn from the TA2 datasets.
