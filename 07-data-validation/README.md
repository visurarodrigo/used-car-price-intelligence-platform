# Stage 07: Data Validation

This stage builds reusable validation checks for new inference payloads before they are passed to the model.

## What It Does

- Reads the cleaned Stage 1 dataset to derive the expected feature schema
- Captures per-feature min, max, median, and binary-state metadata
- Validates incoming payloads for missing fields, unknown fields, numeric coercion, and out-of-range values
- Produces a sample validation report for the bundled API payload

## Outputs

- `outputs/metrics/data_validation_profile.json`
- `outputs/metrics/sample_validation_report.json`

## Run

From project root:

```bash
python 07-data-validation/stage7_data_validation.py
```

## API Integration

The inference API uses the Stage 07 validator to check payloads before prediction and exposes a dedicated `POST /validate` endpoint.