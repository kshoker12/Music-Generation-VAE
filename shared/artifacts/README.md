## Artifact contract (shared)

This folder defines the **artifact contract** between:

- **Kaggle notebooks** (training + export)
- **Local tooling** in `ml/` (validation + preprocessing)
- **Web demo** in `web/` (in-browser inference)

### Source of truth

- `artifact_manifest.schema.json`: JSON Schema for `artifact_manifest.json`

### What the contract is for

- Prevent tokenizer / vocab mismatches
- Prevent silent shape mismatches during inference
- Make the export boundary explicit (what the ONNX file contains)

