# Dataset Viewer

Streamlit viewers for browsing the generated chart dataset, inspecting
per-iteration images, questions, and model evaluation results.

The public dataset viewer is available at
<https://llm-chart-generation.streamlit.app>.

The raw static data served by the web viewer is available from the Biolab file
server at <https://file.biolab.si/llm-chart-generation/>. The root manifest is
at <https://file.biolab.si/llm-chart-generation/manifest.json>.

There are two self-contained variants:

```text
dataset-viewer/
├── local/
│   ├── app.py
│   ├── chart_types.py
│   ├── indexer.py
│   ├── requirements.txt
│   └── README.md
├── web/
│   ├── app_web.py
│   ├── prepare_ftp_bundle.py
│   ├── requirements.txt
│   └── README.md
└── README.md
```

Use the variant-specific README files:

- [local/README.md](local/README.md) for the local filesystem-backed viewer.
- [web/README.md](web/README.md) for Streamlit Community Cloud deployment and
  FTP/static data preparation.
