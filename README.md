# Leaf Image Processing

Course project by Juliano Eleno Silva Pádua and Matheo Sebastian Duarte Davalos.

A Digital Image Processing project on PlantVillage apple leaf images. The
current batch pipeline performs **denoising -> leaf segmentation -> morphology
-> shape/color/texture extraction -> classification**. Sobel and Canny remain
exploratory visual analyses in the notebook and are not classifier inputs.

## Repository Structure

```text
DIP_Project/
  README.md
  requirements.txt
  LICENSE                     # MIT license for original code
  scripts/
    download_data.py          # Optional dataset download and SHA-256 verification
    image_processing.py       # Filtering, segmentation and feature extraction
    run_experiments.py        # Partitions, validation and held-out evaluation
    report_results.py         # Figures derived from saved experiment artifacts
    compare_texture_descriptors.py  # Development-only GLCM/LBP comparison
  notebooks/
    apple_leaf_analysis.ipynb # Demonstration and exploratory analysis
  data/
    samples/                  # Four demonstration images and their provenance
    download_manifest.json    # Pinned source, file sizes and SHA-256 checksums
    raw/                      # Optional full dataset; ignored by Git
  outputs/
    glcm.csv                  # Detailed exploratory GLCM results
    glcm_summary.csv          # Per-class exploratory GLCM summary
```

The notebook keeps the original instructional workflow and imports shared image
processing functions. The reproducible full-dataset experiment is implemented
in `scripts/`; there is no dependency on ImageJ or on a local `leaf_analysis`
package.

## Setup and Execution

Use Python 3.10 and the Python/Jupyter extensions in VS Code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.10 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1` in PowerShell (not verified on Windows).

To run the notebook:

1. Open the project folder and `notebooks/apple_leaf_analysis.ipynb`.
2. Select the Python interpreter inside `.venv`.
3. Restart the kernel and run the notebook from top to bottom.
4. Save the notebook to retain its figures and tables.

For a terminal execution on Linux:

```bash
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/apple_leaf_analysis.ipynb
```

To run a complete experiment, first restore the full dataset, then use a new
output directory. Each protected stage refuses silent replacement:

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
python -m scripts.run_experiments prepare --output outputs/reproduction
python -m scripts.report_results calibration --output outputs/reproduction
python -m scripts.run_experiments extract --output outputs/reproduction
python -m scripts.run_experiments select --output outputs/reproduction
python -m scripts.run_experiments evaluate --output outputs/reproduction
python -m scripts.report_results final --output outputs/reproduction
```

After feature extraction, GLCM and LBP can be compared separately using only
the five development folds, without reopening the held-out test:

```bash
python -m scripts.compare_texture_descriptors \
  --experiment outputs/reproduction \
  --output outputs/texture_comparison.csv
```

## Workflow and Configuration

The notebook demonstration:

1. Loads one image per class and the embedded denoising snapshot.
2. Compares median, bilateral and Gaussian denoising.
3. Applies Sobel and Canny independently for contour inspection.
4. Applies GLCM to configured grayscale ROIs.

The full experiment:

1. Verifies image hashes, groups known duplicates and creates fixed calibration,
   development and test partitions.
2. Applies bilateral filtering.
3. Compares Otsu on HSV saturation, k-means on Lab `a/b`, and the full image.
4. Applies closing, hole filling and largest-component selection to masks.
5. Extracts six shape descriptors, RGB/HSV histograms and moments, six GLCM
   properties and a uniform LBP histogram.
6. Compares feature groups with SVM and Random Forest, selecting parameters only
   in the development folds before evaluating the held-out test.

GLCM uses grayscale intensity, not Canny's binary map. Masked GLCM counts only
pairs inside the selected region, while LBP uses only complete internal
neighborhoods. The additional descriptor comparison separates GLCM and LBP on
the existing development folds.

The notebook setup cell contains filter selection, Gaussian parameters, Canny
thresholds, ROI coordinates and GLCM levels/distances/angles.
`MOSTRAR_HISTORICOS=False` leaves the separate NLM/BM3D functions available
without executing those optional historical comparisons.

## Interpretation Limits

- The frozen denoising snapshot is dated 2026-09-21 16:39 UTC. Its
  MSE/PSNR/SSIM values concern synthetic noise and do not establish the best
  filter for classification.
- The 64x64 notebook ROIs are exploratory, not annotated lesions. Their GLCM
  properties do not diagnose disease.
- Sobel and Canny show intensity transitions, not validated leaf or lesion masks.
- Otsu and k-means masks have no manually annotated reference for IoU/Dice.
- Image grouping detects exact and close copies but does not identify every
  photograph of the same biological leaf or acquisition session.
- Performance on the controlled collection does not demonstrate agronomic
  diagnosis or generalization to field photographs.

## Exploratory GLCM Results

The committed [summary CSV](outputs/glcm_summary.csv) contains the following
values (rounded to four decimals). Each row represents **one image and one
manually selected 64×64 ROI**, with 32 gray levels. Descriptors are averaged over
12 distance/angle combinations: distances 1, 2 and 4 pixels; angles 0°, 45°, 90°
and 135°. These are not averages over multiple images per class.

| Example class (n = 1 each) | Contrast | Dissimilarity | Homogeneity | Energy | ASM | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Apple scab | 3.0042 | 1.0032 | 0.6471 | 0.2450 | 0.0602 | 0.6416 |
| Black rot | 29.7860 | 3.7955 | 0.2986 | 0.0761 | 0.0058 | 0.7531 |
| Cedar apple rust | 0.5107 | 0.3568 | 0.8367 | 0.2862 | 0.0827 | 0.9188 |
| Healthy | 2.8014 | 0.9490 | 0.6632 | 0.2240 | 0.0503 | 0.6653 |

For these particular ROIs, black rot has the highest contrast (29.7860), while
cedar apple rust has the highest homogeneity (0.8367). These observations
describe selected regions; they do not establish disease-specific signatures,
classification accuracy or generalization. The [detailed CSV](outputs/glcm.csv)
provides all 48 descriptor rows.

## Images and Dataset

Only the **four demonstration images** in `data/samples/` are versioned as
dataset inputs. They are sufficient for the notebook demonstration. The
[sample notes](data/samples/README.md) document their provenance and SHA-256
checksums. The full dataset is restored to `data/raw/`, ignored by Git, and is
required for the batch classification experiment.

To restore the unaugmented apple subset using Python's standard library:

```bash
python scripts/download_data.py
python scripts/download_data.py --verify-only
```

Use `--subset with` for augmented images or `--subset all` for both subsets.
The script downloads a ZIP of the repository's pinned historical snapshot and
extracts only requested images after verifying sizes and SHA-256 hashes against
[the manifest](data/download_manifest.json). Already valid files are reused.
Use `--archive /path/to/snapshot.zip` to reuse a downloaded ZIP offline.

| Apple class | Without augmentation | With augmentation |
| --- | ---: | ---: |
| Healthy | 1,645 | 1,645 |
| Apple scab | 630 | 1,000 |
| Black rot | 621 | 1,000 |
| Cedar apple rust | 275 | 1,000 |
| **Total** | **3,171** | **4,645** |

These counts and checksums identify the project's preserved dataset, not an
independently verified copy of the upstream Mendeley archives. The reference
source is [Mendeley Data, version 1](https://data.mendeley.com/datasets/tywbtsjrjv/1),
DOI `10.17632/tywbtsjrjv.1`.

Approximately 111 MiB of images remain in earlier Git commits. Removing them
from the current tree prevents further routine tracking but does **not** shrink
existing history or normal full clones. A shallow clone avoids downloading that
history. Any future history rewrite must first migrate the pinned download to a
durable source and update the manifest.

## Attribution

Dataset distribution: ARUN PANDIAN J; GEETHARAMANI GOPAL (2019), *Data for:
Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural
Network*, Mendeley Data, V1, DOI `10.17632/tywbtsjrjv.1`.

PlantVillage background: Hughes and Salathe (2015),
[*An open access repository of images on plant health to enable the development
of mobile disease diagnostics*](https://arxiv.org/abs/1511.08060).

## License

Original project code, including code cells in the notebook and scripts, is
licensed under the [MIT License](LICENSE). Third-party packages retain their own
licenses. Dataset images, including images embedded in notebook figures, are
**not covered by the project's MIT license**.

The referenced [Mendeley distribution](https://data.mendeley.com/datasets/tywbtsjrjv/1)
is marked **CC0 1.0**. The precise correspondence between inherited local images
and that distribution has not been independently verified; see the
[sample provenance notes](data/samples/README.md). This project does not grant
additional rights to third-party images.
