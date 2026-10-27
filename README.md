# Advanced Process Data Analytics Course for Bioprocessing

Jupyter notebooks of the DataHow teaching course. Every notebook can be run in Google Colab (nothing to install) or locally in a Python environment managed with [uv](https://docs.astral.sh/uv/).

## Course structure

Part 1

* `00_Process_Introduction` – the in silico fed-batch cell culture simulator and the design of experiments used in the course
* `01_PCA` – principal component analysis, OWU and BWU unfolding
* `02_PLS` – linear regression, model diagnostics and partial least squares
* `03_GP` – Gaussian processes from conditional probabilities and kernels

Part 2

* `04_Hybrid_Model` – hybrid models combining mass balances with Gaussian processes
* `05_Bayesian_Optimization` – acquisition functions and Bayesian optimization
* `06_Ensembles` – model variation, ensembles and bootstrap prediction intervals
* `07_TransferLearning` – transfer learning between two similar processes

The notebooks are in `scripts/`. PDF exports of the executed notebooks are in `pdf/`.

## Run in Google Colab

1. Open https://colab.research.google.com/ and choose File > Open notebook (Ctrl/Cmd + O).
2. Select the GitHub tab, enter `https://github.com/DataHow/analytics-course-scripts` and search.
3. Keep the `main` branch and open a notebook from `scripts/`, starting with `scripts/00_Process_Introduction.ipynb`.
4. Run the first cell. It clones this repository into the Colab session and installs the extra packages.

## Run locally

1. Install uv: https://docs.astral.sh/uv/getting-started/installation/
2. Clone the repository and create the environment (Python 3.12 is downloaded by uv if needed):

   ```bash
   git clone https://github.com/DataHow/analytics-course-scripts.git
   cd analytics-course-scripts
   uv sync
   ```

3. Start JupyterLab and open a notebook from `scripts/`:

   ```bash
   uv run jupyter lab scripts
   ```

   In VS Code, open the folder and select the interpreter in `.venv` as the notebook kernel.

## Export the notebooks to PDF

```bash
uv sync --group export
uv run playwright install chromium
uv run python tools/export_pdf.py
```

Plotly figures are rendered to images with kaleido, which needs a Chrome or Chromium browser installed on the machine. The PDFs are written to `pdf/`.

## Repository layout

* `scripts/` – the course notebooks
* `scripts/modules/` – the process simulator and plotting helpers used by the notebooks
* `scripts/datasets/` – pre-generated train and test datasets (`owu*.csv`) and the food consumption dataset
* `scripts/generated/` – datasets generated while running the notebooks (not versioned)
* `interactions/` – interactive widget versions of the lectures, made for Google Colab
* `tools/export_pdf.py` – PDF export script
