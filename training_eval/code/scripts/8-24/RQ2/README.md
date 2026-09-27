# SciRepEval full-corpus RQ2

Train BGE-M3 or Qwen3-Embedding-8B on the two SciRepEval synthetic JSONL
files, then evaluate the trained model with full-corpus retrieval at top 10.

```bash
# Train/evaluate one family. Choose GPUs that are free on the current host.
CUDA_VISIBLE_DEVICES=4,5 bash run.sh bge-m3
CUDA_VISIBLE_DEVICES=4,5 bash run.sh qwen3-8b

# Run both families sequentially.
CUDA_VISIBLE_DEVICES=4,5 bash run.sh all
```

The default training files are taken from `SYN_DATA_DIR`; override it when
using another synthetic-data directory. The evaluator records `ndcg_cut_10`,
`recall_10`, `judged_at_10`, and `positive_hit_rate_at_10`. These are
full-corpus diagnostic metrics, not the official ten-candidate SciRepEval
Search score.
