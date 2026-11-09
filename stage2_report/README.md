# Stage II Module 3 — crash understanding report

A second Qwen3.5-27B QLoRA adapter. The vision tower stays frozen; LoRA is on the language layers only. This adapter is trained separately from the spatial adapter in `../stage2_spatial_vlora/`.

Training schedule in `lora_finetune/run_train.sh`:

1. One epoch at learning rate \(1\times10^{-4}\) on the mixed train split, to learn the report schema.
2. Up to 10 epochs at \(2\times10^{-5}\) on the gold subset, with early stopping.

```bash
cd lora_finetune
python prepare_data.py
bash run_train.sh
python infer.py --adapter runs/stage2_gold_refine --test_jsonl data/test.jsonl --out runs/stage2_gold_refine/preds_test.jsonl
python eval_preds.py --preds runs/stage2_gold_refine/preds_test.jsonl
```

WHEN / WHERE / WHAT priors are built in `prompts.py`. Frame offsets for the 20-frame window are in `frame_strategies.py` (`pre13_post7`).
