# Web Dataset Viewer

Use `app_web.py` for Streamlit Community Cloud. It is intentionally light: it
loads only the global manifest and chart index at startup, shows thumbnails
from static URLs, and fetches one chart's metadata/results/images only when a
chart detail page is opened.

The public deployment is available at
<https://llm-chart-generation.streamlit.app>.

The raw static data used by the public viewer is available from the Biolab file
server at <https://file.biolab.si/llm-chart-generation/>. It includes the
manifest, chart index, thumbnails, per-chart metadata, per-chart model results,
and chart images.

## Streamlit Deployment

Streamlit Community Cloud entrypoint:

```text
dataset-viewer/web/app_web.py
```

The web app defaults to:

```text
https://file.biolab.si/llm-chart-generation/manifest.json
```

To point it at another static data location, set a Streamlit secret or
environment variable named `REMOTE_MANIFEST_URL`.

## Generate Static Data

Generate the static bundle from the repository root:

```bash
.venv/bin/python chart-generation/dataset-viewer/web/prepare_ftp_bundle.py \
  --output /tmp/chart-viewer-data \
  --base-url https://file.biolab.si/llm-chart-generation \
  --clean
```

The global grid order is randomly shuffled with fixed seed `42` by default, so
the order is stable across deployments. Change the seed or disable shuffling:

```bash
--shuffle-seed 123
--shuffle-seed none
```

Detail-view images are converted to JPEG and capped at 1400 px wide by default.
Adjust this if needed:

```bash
.venv/bin/python chart-generation/dataset-viewer/web/prepare_ftp_bundle.py \
  --output /tmp/chart-viewer-data \
  --base-url https://file.biolab.si/llm-chart-generation \
  --detail-image-width 1200 \
  --detail-jpeg-quality 85 \
  --clean
```

The generated server layout is:

```text
llm-chart-generation/
├── manifest.json
├── charts.jsonl.gz
├── thumbnails/
│   └── <dataset-id>/
│       └── <chart-id>.jpg
├── datasets/
│   ├── <dataset-id>.json
│   └── <dataset-id>.charts.jsonl
└── charts/
    └── <dataset-id>/
        └── <chart-id>/
            ├── manifest.json
            ├── metadata.json
            ├── results.jsonl.gz
            └── images/
                └── <iteration-image>.png
```

Each chart directory is self-contained:

```text
manifest.json
metadata.json
results.jsonl.gz
images/
```

## Upload Checklist

1. Generate the static bundle:

   ```bash
   .venv/bin/python chart-generation/dataset-viewer/web/prepare_ftp_bundle.py \
     --output /tmp/chart-viewer-data \
     --base-url https://file.biolab.si/llm-chart-generation \
     --clean
   ```

2. Inspect the generated bundle:

   ```bash
   du -sh /tmp/chart-viewer-data
   find /tmp/chart-viewer-data -maxdepth 2 -type f | sort | head
   ```

3. Upload the contents to the web server. The public URL is
   `https://file.biolab.si/llm-chart-generation/`, but `rsync` needs the SSH
   target and filesystem path:

   ```bash
   rsync -avz --delete --progress \
     /tmp/chart-viewer-data/ \
     USER@file.biolab.si:/PATH/TO/llm-chart-generation/
   ```

   Replace `USER` and `/PATH/TO/llm-chart-generation/` with the server account
   and directory that map to the public URL.

4. Verify the upload:

   ```text
   https://file.biolab.si/llm-chart-generation/manifest.json
   https://file.biolab.si/llm-chart-generation/charts.jsonl.gz
   https://file.biolab.si/llm-chart-generation/charts/<dataset-id>/<chart-id>/metadata.json
   https://file.biolab.si/llm-chart-generation/charts/<dataset-id>/<chart-id>/results.jsonl.gz
   ```
