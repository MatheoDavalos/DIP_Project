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

## Optional Complete Apple Dataset

Only these four sample images are tracked. The complete apple subsets belong in
`data/raw/`, which is ignored by Git. From the repository root, run:

```bash
python scripts/download_data.py
python scripts/download_data.py --subset all
python scripts/download_data.py --subset all --verify-only
```

The default restores 3,171 unaugmented images; `--subset all` restores those plus
4,645 augmented images. The script uses a pinned historical project snapshot,
with exact file sizes and SHA-256 hashes in `data/download_manifest.json`.
These checksums establish the identity of the preserved project inputs, not
identity with an independently downloaded Mendeley archive. Full Git history
still contains the old dataset; see the root README for the shallow-clone option.

The notebook continues to read only the four files in `data/samples/`.
Third-party images are excluded from the project's MIT software license.
