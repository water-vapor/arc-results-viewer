# ARC Results Viewer

Minimal Streamlit viewer for ARC challenge results. Real data lives in
`results_json/` (one JSON per model) and the app reads a preprocessed parquet.

## Setup

```bash
pip install -r requirements.txt
```

## Preprocess data

Turn the raw predictions in `results_json/` into `arc_results_processed.parquet`:

```bash
python preprocess_data.py
```

## Run the app

Launch the viewer after preprocessing finishes:

```bash
streamlit run app.py
```

Use the sidebar to point at a different parquet file, adjust the SQL filter, or change the table height.
