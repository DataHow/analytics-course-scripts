"""Export the course notebooks to PDF.

Usage (from the repository root, after `uv sync --group export`):

    uv run python tools/export_pdf.py              # all notebooks in scripts/
    uv run python tools/export_pdf.py scripts/03_GP.ipynb

Steps per notebook:
1. Interactive Plotly outputs are rendered to PNG with kaleido (needs a Chrome
   or Chromium installation, see https://github.com/plotly/kaleido).
2. The notebook is converted to HTML; code cells tagged "hide-input" are
   exported without their source.
3. The HTML is printed to PDF with Playwright. The system Google Chrome is
   used when available, otherwise Playwright's own Chromium
   (install once with `uv run playwright install chromium`).

The PDFs are written to the folder pdf/ next to scripts/.
"""
import base64
import json
import sys
import tempfile
from pathlib import Path

import nbformat
import plotly.graph_objects as go
import plotly.io as pio
from nbconvert import HTMLExporter
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright
from traitlets.config import Config

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "pdf"
PLOTLY_MIME = "application/vnd.plotly.v1+json"

PRINT_CSS = """
<style>
@page { size: A4; margin: 14mm 12mm; }
body { font-size: 11px; }
.jp-Notebook { padding: 0 !important; }
.jp-OutputArea-child, .jp-CodeMirrorEditor { page-break-inside: avoid; }
.jp-RenderedHTMLCommon h1, .jp-RenderedHTMLCommon h2, .jp-RenderedHTMLCommon h3 { page-break-after: avoid; }
.jp-RenderedHTMLCommon h1 { font-size: 1.8em; }
pre, .jp-RenderedText pre { white-space: pre-wrap !important; word-break: break-word; }
img { max-width: 100% !important; height: auto !important; }
.jp-RenderedHTMLCommon table { font-size: 9px; }
</style>
"""


def plotly_outputs_to_png(nb, scale=2):
    """Replace every Plotly JSON output with a static PNG rendering."""
    n = 0
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if PLOTLY_MIME not in data:
                continue
            spec = data[PLOTLY_MIME]
            if isinstance(spec, str):
                spec = json.loads(spec)
            fig = go.Figure({k: spec.get(k) for k in ("data", "layout", "frames") if spec.get(k) is not None})
            width = fig.layout.width or 900
            height = fig.layout.height or 500
            png = pio.to_image(fig, format="png", width=width, height=height, scale=scale)
            output["data"] = {
                "image/png": base64.b64encode(png).decode("ascii"),
                "text/plain": "<Figure>",
            }
            output["metadata"] = {"image/png": {"width": width, "height": height}}
            n += 1
    return n


def notebook_to_html(nb, name):
    config = Config()
    config.TagRemovePreprocessor.remove_input_tags = {"hide-input"}
    config.TagRemovePreprocessor.enabled = True
    config.HTMLExporter.preprocessors = ["nbconvert.preprocessors.TagRemovePreprocessor"]
    config.HTMLExporter.exclude_input_prompt = True
    config.HTMLExporter.exclude_output_prompt = True
    exporter = HTMLExporter(config=config)
    body, _ = exporter.from_notebook_node(nb, resources={"metadata": {"name": name}})
    return body.replace("</head>", PRINT_CSS + "</head>", 1)


def html_to_pdf(html_path: Path, pdf_path: Path, browser):
    page = browser.new_page()
    page.goto(html_path.as_uri(), wait_until="networkidle")
    # give MathJax time to typeset the formulas
    page.wait_for_timeout(1500)
    page.pdf(path=str(pdf_path), format="A4", print_background=True, prefer_css_page_size=True)
    page.close()


def launch_browser(p):
    try:
        return p.chromium.launch(channel="chrome")
    except PlaywrightError:
        return p.chromium.launch()


def export_all(paths):
    OUT_DIR.mkdir(exist_ok=True)
    with sync_playwright() as p, tempfile.TemporaryDirectory() as tmp:
        browser = launch_browser(p)
        for path in paths:
            nb = nbformat.read(path, as_version=4)
            n = plotly_outputs_to_png(nb)
            html_path = Path(tmp) / f"{path.stem}.html"
            html_path.write_text(notebook_to_html(nb, path.stem), encoding="utf-8")
            pdf_path = OUT_DIR / f"{path.stem}.pdf"
            html_to_pdf(html_path, pdf_path, browser)
            print(f"{path.name}: {n} figures rendered, written to {pdf_path.relative_to(ROOT)} ({pdf_path.stat().st_size / 1e6:.1f} MB)")
        browser.close()


if __name__ == "__main__":
    args = [Path(a).resolve() for a in sys.argv[1:]] or sorted((ROOT / "scripts").glob("*.ipynb"))
    export_all(args)
