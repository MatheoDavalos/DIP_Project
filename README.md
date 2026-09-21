# Leaf Image Processing

A notebook-based Digital Image Processing course project exploring denoising,
frequency analysis and edge detection on apple leaf images from PlantVillage.

The current scope is a **four-image demonstration**, one image per apple class.
It does not process the complete dataset, train a classifier or diagnose disease.
The main implementation lives in a single Jupyter notebook. This README is in
English; the notebook retains its Portuguese explanations for the course.

## Project Structure

```text
leaf-image-processing/
  README.md
  requirements.txt
  .gitignore
  notebooks/
    apple_leaf_analysis.ipynb      # Main workflow
  data/
    raw/
      without/                    # Complete unaugmented apple subset
      with/                       # Complete augmented apple subset
    samples/
      README.md                   # Dataset attribution and sample selection
      manifest.json               # File sizes and SHA-256 checksums
      Apple___healthy/
      Apple___Apple_scab/
      Apple___Black_rot/
      Apple___Cedar_apple_rust/
  outputs/
    tuning_snapshot.json          # Frozen tuning parameters and metrics
    tuning_summary.csv            # Tabular tuning metrics, including partial status
    edge_stability.csv            # Current edge stability results
```

The following directories are local-only and excluded from Git:

- `local/`: study reports, earlier notebooks, historical results and tests
  awaiting further organization.
- `*.executed.ipynb`: temporary executed copies, if generated from the terminal.
- `.venv/` and editor configuration: machine-specific environment and settings.

These local-only folders are not included when cloning. The complete apple
dataset under `data/raw/` IS included. The main notebook contains its parameter
snapshot and reads only the four bundled demonstration images in `data/samples/`.

## Setup

The project was checked on Linux with Python **3.10.12**. The direct dependency
versions in `requirements.txt` match the working environment. Transitive
dependencies are resolved by pip; this is not a complete environment lockfile.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/MatheoDavalos/leaf-image-processing.git
cd leaf-image-processing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name leaf-image-processing --display-name "Python (Leaf Image Processing)"
```

On Windows, use `py -3.10 -m venv .venv` to create the environment and
`.venv\Scripts\Activate.ps1` in PowerShell to activate it. The remaining Python
commands are the same. Windows execution has not been verified.

### Run in VS Code

1. Open the repository folder in VS Code with the Python and Jupyter extensions.
2. Open [`notebooks/apple_leaf_analysis.ipynb`](notebooks/apple_leaf_analysis.ipynb).
3. Select **Python (Leaf Image Processing)**, or the interpreter inside `.venv`.
4. Restart the kernel and run all cells from top to bottom.

The notebook finds the project root when launched from the root or a directory
inside it. No absolute machine-specific data path is required.

### Run in JupyterLab

JupyterLab is an optional interface, installed separately:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/apple_leaf_analysis.ipynb
```

Select the same kernel and run all cells. BM3D can take longer than the other
filters; execution time depends on your CPU.

### Execute from the Terminal

From the repository root, using the activated environment:

```bash
mkdir -p outputs
python -m jupyter nbconvert --to notebook --execute notebooks/apple_leaf_analysis.ipynb --ExecutePreprocessor.kernel_name=leaf-image-processing --ExecutePreprocessor.timeout=600 --output apple_leaf_analysis.executed.ipynb --output-dir outputs
```

The executed copy is saved locally in `outputs/` and ignored by Git. The public
notebook includes its figures and tables, so results can also be viewed without
running Python. To update that single public notebook after terminal execution:

```bash
cp outputs/apple_leaf_analysis.executed.ipynb notebooks/apple_leaf_analysis.ipynb
```

Running and saving directly in VS Code or JupyterLab updates the main notebook
without this copy step.

## Notebook Workflow

1. Load one original image per class and the embedded tuning snapshot.
2. Compare median, bilateral, Non-Local Means and BM3D filters using previously
   selected parameters. Gaussian filtering is a separate manual comparison.
3. Show each filter's effect on the same four original images.
4. Record numerical, visual and final filter choices separately.
5. Inspect FFT spectra, demonstrate a Gaussian low-pass candidate and reconstruct
   with IFFT. Optional notch filtering requires manually specified peaks and a
   justification; by default, the selected frequency mask is the identity.
6. Apply Sobel and Canny independently to the same preprocessed images.
7. Measure edge-map stability over five small Gaussian perturbations using IoU
   and a one-pixel-tolerant F1 score.

The existing snapshot was recorded on **2026-09-21 at 16:39 UTC**. Noise levels
5 and 10 contain all four tuned methods; level 15 is partial and is excluded
from selection. The default is level 10. The notebook does not rerun tuning or
read the original local results file.

### Configuration

Edit the configuration cells near the start, then restart and run all cells:

| Setting | Purpose |
| --- | --- |
| `SIGMA_PARAMETROS` | Select a complete tuning snapshot: 5 or 10. |
| `FILTRO_VISUAL`, `JUSTIFICATIVA_VISUAL` | Record a human visual preference. |
| `FILTRO_FINAL`, `JUSTIFICATIVA_FINAL` | Record the final choice; `None` uses the numerical winner provisionally. |
| `PARAMETROS_GAUSSIANO` | Set the manual spatial Gaussian comparison. |
| `APLICAR_FILTRO_FREQUENCIAL`, `PICOS_NOTCH`, `JUSTIFICATIVA_FREQUENCIAL` | Enable and justify optional notch filtering. |
| `CORTE_GAUSSIANO_FFT` | Set the illustrative low-pass candidate cutoff. |
| `LIMIAR_SOBEL`, `LIMIARES_CANNY` | Set shared edge thresholds. |
| `SIGMA_ESTABILIDADE`, `REPETICOES_ESTABILIDADE`, `SEMENTE_BORDAS` | Configure the edge stability experiment. |

The in-memory outputs for later work are `IMAGENS_PRE_PROCESSADAS`,
`RESULTADOS_BORDAS` and `TABELA_ESTABILIDADE`.

The final cell also overwrites three unified result files in `outputs/`:
`tuning_snapshot.json`, `tuning_summary.csv` and `edge_stability.csv`. The first
two reproduce the historical tuning snapshot, including its partial sigma 15
status; the edge stability table is recomputed by the current execution.
Figures remain in the main notebook instead of duplicated executed notebooks.

### Interpretation Limits

- MSE, PSNR and SSIM are frozen **tuning results**, not quality measurements of
  the four displayed images and not a final generalization assessment.
- Demonstration filters operate directly on the original images. Transferring
  parameters tuned with synthetic noise still requires visual assessment.
- FFT reconstruction error measures numerical consistency, not image quality.
- Edge IoU/F1 measures repeatability under perturbations applied after denoising.
  There are no annotated ground-truth contours; this is not segmentation accuracy.
- Four examples do not support conclusions about the complete dataset.

## Images and Dataset

The repository includes the **complete apple subset**, both unaugmented and
augmented versions, so cloning also downloads the data for future experiments.
The notebook still processes only the four selected images in `data/samples/`
(about 56 KB), which are unmodified copies of files in `data/raw/without/`.
The small copies make the current demonstration's inputs explicit and stable.

The complete dataset contains approximately 111 MiB of image data. These are
ordinary Git files, so Git LFS is not required. Avoid repeatedly replacing image
binaries: Git history retains old versions and can grow over time.

The [sample notes](data/samples/README.md) include filenames, provenance details,
checksums and download instructions. A reference distribution is available from
[Mendeley Data, version 1](https://data.mendeley.com/datasets/tywbtsjrjv/1),
DOI `10.17632/tywbtsjrjv.1`. No separate download is required after cloning this
repository. To obtain an independent copy from the source, download the **without
augmentation** archive and keep all images in the four `Apple___*` directories
under `data/raw/without/`; use `data/raw/with/` for the augmented archive.
The current notebook reads only `data/samples/`; the larger dataset does not
trigger a batch run.

The original local workspace already contains the full apple class counts below.
These files were preserved during reorganization, so no second download was
needed on the author's computer. A fresh clone includes both apple subsets.

| Apple class | Without augmentation | With augmentation |
| --- | ---: | ---: |
| Healthy | 1,645 | 1,645 |
| Apple scab | 630 | 1,000 |
| Black rot | 621 | 1,000 |
| Cedar apple rust | 275 | 1,000 |
| **Total** | **3,171** | **4,645** |

These are local file counts, not a checksum verification against the downloaded
archives. Filesystem allocation can occupy more space than the image bytes. For future
experiments, retain the unaugmented base and define data splits before generating
additional augmented variants.

## Initialize and Publish a Repository

Cloning already initializes Git and configures `origin`. The following steps
are for a fresh local copy that does **not** yet have a Git repository:

```bash
git init -b main
git config user.name "Your Name"
git config user.email "your-email@example.com"
git add README.md requirements.txt .gitignore notebooks data outputs
git diff --cached --stat
git commit -m "Initialize Leaf Image Processing"
```

Install the [GitHub CLI](https://cli.github.com/) and authenticate through the
browser, then create a new public repository in your account:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh repo create leaf-image-processing --public --source=. --remote=origin --push
```

Run the creation command only once, when the remote repository does not exist.
For subsequent changes, run and save the main notebook to keep its figures,
tables and exported results synchronized, then commit the intended changes:

```bash
git status --short
git add notebooks/apple_leaf_analysis.ipynb outputs
git diff --cached --stat
git commit -m "Describe the change"
git push
```

Add any other intentionally changed files by name. Study documents, historical
experiments and tests stay local through `.gitignore`. The apple datasets and
current unified results are intentionally tracked.

## Attribution

Dataset distribution: ARUN PANDIAN J; GEETHARAMANI GOPAL (2019), *Data for:
Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural
Network*, Mendeley Data, V1, DOI `10.17632/tywbtsjrjv.1`.

PlantVillage background: Hughes and Salathe (2015),
[*An open access repository of images on plant health to enable the development
of mobile disease diagnostics*](https://arxiv.org/abs/1511.08060).

Third-party images and packages retain their own terms. No separate software
license has been selected for this course project.
