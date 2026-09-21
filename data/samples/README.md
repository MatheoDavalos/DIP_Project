# Sample Images

Four unmodified images were copied from the project's existing local
`without/` apple subset. These are exactly the filenames already selected by the
main notebook, not a newly sampled dataset.

| Class | Filename |
| --- | --- |
| Healthy | `Apple___healthy/image (1000).JPG` |
| Apple scab | `Apple___Apple_scab/image (101).JPG` |
| Black rot | `Apple___Black_rot/image (101).JPG` |
| Cedar apple rust | `Apple___Cedar_apple_rust/image (101).JPG` |

The notebook describes them as the fourth lexicographically sorted image in
each class, separate from the first three used for tuning. They are visual
examples, not a representative evaluation set. See `manifest.json` for each
file's exact size and SHA-256 checksum; no resizing or recompression was applied.

## Dataset Reference

The existing notebooks identify the images as Apple / PlantVillage. A reference
download with the corresponding apple classes and augmented/unaugmented layout
is [Mendeley Data, version 1](https://data.mendeley.com/datasets/tywbtsjrjv/1):

ARUN PANDIAN J; GEETHARAMANI GOPAL (2019), *Data for: Identification of Plant Leaf
Diseases Using a 9-layer Deep Convolutional Neural Network*, Mendeley Data, V1.
DOI: `10.17632/tywbtsjrjv.1`.

The Mendeley record lists **CC0 1.0**. The local workspace did not contain an
original download receipt or archive checksum, so the correspondence of these
local files to that specific distribution has not been independently verified.
The manifest records the actual local files used here, not a verified upstream
archive identity.

Background: Hughes and Salathe (2015),
[PlantVillage paper](https://arxiv.org/abs/1511.08060).

## Complete Apple Dataset

Both complete apple subsets are included in this repository under `data/raw/`,
so a normal clone downloads them. To obtain another copy from the source:

1. Open the Mendeley record and download the **without augmentation** archive.
2. Extract all images from the four `Apple___*` class folders into `data/raw/without/`.
3. Keep each class folder and original filename unchanged.
4. Keep augmented images, if needed later, separately under `data/raw/with/`.

Both raw folders are tracked by Git. Their presence does not change which
images the main notebook reads. Its default `PASTA_IMAGENS` remains `data/samples`.
All processing in the current notebook is limited to the four selected files.

The repository contains 3,171 unaugmented apple images and 4,645 images in the
augmented version. The complete apple subset is available for later experiments;
the four demonstration copies remain the only inputs to the main notebook.
