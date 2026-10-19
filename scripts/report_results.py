"""Figuras e relatório derivados dos arquivos do experimento, sem reajustar modelos."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.image_processing import (filtro_bilateral, filtro_gaussiano, filtro_mediana,
                                      segment_leaf)
from scripts.run_experiments import ROOT, read_image, read_partitions

NAMES = {"Apple___Apple_scab": "Sarna", "Apple___Black_rot": "Podridão negra",
         "Apple___Cedar_apple_rust": "Ferrugem", "Apple___healthy": "Saudável"}


def calibration_figures(output):
    partitions = read_partitions(output)
    samples = partitions[partitions.split == "calibration"].groupby("class").head(6)
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for method in ("original", "otsu", "kmeans"):
        fig, axes = plt.subplots(4, 6, figsize=(15, 10), layout="constrained")
        for ax, (_, row) in zip(axes.flat, samples.iterrows()):
            image = filtro_bilateral(read_image(row.path))
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if method != "original":
                try:
                    mask = segment_leaf(image, method)
                    ax.contour(mask, levels=[.5], colors=["yellow"], linewidths=1)
                    ax.imshow(np.dstack((np.zeros_like(mask), np.zeros_like(mask), np.ones_like(mask), (~mask)*.35)))
                except ValueError as error:
                    ax.text(.5, .5, str(error), transform=ax.transAxes, ha="center", wrap=True, color="red")
            ax.set_title(f"{NAMES[row['class']]} | {Path(row.path).name}", fontsize=8)
            ax.axis("off")
        fig.savefig(directory / f"calibration_{method}.png", dpi=130)
        plt.close(fig)
    samples = samples.groupby("class").head(1)
    fig, axes = plt.subplots(4, 4, figsize=(11, 11), layout="constrained")
    for row_index, (_, row) in enumerate(samples.iterrows()):
        image = read_image(row.path)
        for column, (name, filtered) in enumerate((('Original', image), ('Mediana', filtro_mediana(image)),
                                                  ('Gaussiano', filtro_gaussiano(image)), ('Bilateral', filtro_bilateral(image)))):
            axes[row_index, column].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            axes[row_index, column].set_title(f"{NAMES[row['class']]} | {name}", fontsize=9)
            axes[row_index, column].axis("off")
    fig.savefig(directory / "filters.png", dpi=130)
    plt.close(fig)


    examples = partitions[partitions.split == "calibration"].groupby("class").nth(3)
    fig, axes = plt.subplots(4, 3, figsize=(8, 10), layout="constrained")
    for row_index, (_, row) in enumerate(examples.iterrows()):
        image = filtro_bilateral(read_image(row.path))
        for column, method in enumerate(("original", "otsu", "kmeans")):
            ax = axes[row_index, column]
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if method != "original":
                mask = segment_leaf(image, method)
                ax.contour(mask, levels=[.5], colors=["yellow"], linewidths=1)
                ax.imshow(np.dstack((np.zeros_like(mask), np.zeros_like(mask), np.ones_like(mask), (~mask)*.35)))
            ax.set_title(f"{NAMES[row['class']]} | {method}", fontsize=9)
            ax.axis("off")
    fig.savefig(directory / "segmentation_comparison.png", dpi=150)
    plt.close(fig)


def final_figures(output):
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    metrics = pd.read_csv(output / "test_metrics.csv")
    frozen = json.loads((output / "frozen.json").read_text())
    matrices = json.loads((output / "confusion_matrices.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, classifier in zip(axes, ("svm", "rf")):
        setting = max((e for e in frozen['experiments'] if e['classifier'] == classifier), key=lambda e: e['validation_f1'])
        data = matrices[setting['experiment']]
        matrix = np.array(data['matrix'])
        ax.imshow(matrix, cmap='Blues')
        for (y, x), value in np.ndenumerate(matrix):
            ax.text(x, y, str(value), ha='center', va='center', color='white' if value > matrix.max()/2 else 'black')
        ax.set(xticks=range(4), yticks=range(4), xlabel='Predição', ylabel='Classe real',
               title=classifier.upper() + ' | escolhido na validação')
        ax.set_xticklabels([NAMES[c] for c in data['classes']], rotation=30, ha='right')
        ax.set_yticklabels([NAMES[c] for c in data['classes']])
    fig.savefig(directory / 'confusion.png', dpi=160)
    plt.close(fig)
    predictions = pd.read_csv(output / 'predictions.csv')
    winner = predictions[predictions.experiment == frozen['winner']]
    errors = winner[winner.truth != winner.prediction].groupby("truth", sort=True).head(2)
    if errors.empty:
        errors = winner.head(8)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), layout='constrained')
    winner_setting = next(e for e in frozen['experiments'] if e['experiment'] == frozen['winner'])
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, errors.itertuples()):
        image = filtro_bilateral(read_image(row.path))
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if winner_setting['method'] != 'unmasked':
            try:
                mask = segment_leaf(image, winner_setting['method'])
                ax.contour(mask, levels=[.5], colors=['yellow'], linewidths=1)
            except ValueError:
                pass  # A falha já está registrada em segmentation.csv; manter a imagem no painel.
        ax.set_title(f"Real: {NAMES[row.truth]}\nPred.: {NAMES[row.prediction]}\n{Path(row.path).name}", fontsize=9)
    fig.savefig(directory / 'errors.png', dpi=150)
    plt.close(fig)
    diagnostics = pd.read_csv(output / 'segmentation.csv').merge(read_partitions(output), on='path')
    diagnostics['failed'] = diagnostics.error.notna()
    coverage = diagnostics.groupby(['split', 'class', 'method']).agg(images=('path', 'size'), failed=('failed', 'sum'))
    coverage['coverage'] = 1 - coverage.failed / coverage.images
    coverage.to_csv(output / 'coverage.csv')
    return metrics, frozen, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('calibration', 'final'))
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/experiment')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    if args.stage == 'calibration':
        calibration_figures(args.output)
    else:
        final_figures(args.output)


if __name__ == '__main__':
    main()
