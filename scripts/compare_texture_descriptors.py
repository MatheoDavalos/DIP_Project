"""Compara GLCM e LBP separadamente usando apenas a validação existente."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

from scripts.run_experiments import ROOT, load_features, model_grid


def texture_columns(frame, descriptor):
    lbp = [column for column in frame if column.startswith("texture_lbp_")]
    glcm = [column for column in frame
            if column.startswith("texture_") and column not in lbp]
    columns = {"glcm": glcm, "lbp": lbp}[descriptor]
    if not columns:
        raise ValueError(f"No {descriptor.upper()} columns found.")
    return columns


def compare(experiment, destination):
    frame = load_features(experiment)
    frame = frame.loc[(frame.split == "development") &
                      (frame.method == "kmeans")].reset_index(drop=True)
    cv = [(np.flatnonzero(frame.fold != fold), np.flatnonzero(frame.fold == fold))
          for fold in range(5)]
    results = []
    for descriptor in ("glcm", "lbp"):
        columns = texture_columns(frame, descriptor)
        for classifier in ("svm", "rf"):
            estimator, grid = model_grid(classifier)
            search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro",
                                  n_jobs=1, refit=False, error_score="raise")
            search.fit(frame[columns], frame["class"])
            best = search.best_index_
            results.append({
                "descriptor": descriptor.upper(),
                "classifier": classifier.upper(),
                "features": len(columns),
                "params": json.dumps(search.best_params_, sort_keys=True),
                "mean_f1": search.cv_results_["mean_test_score"][best],
                "std_f1": search.cv_results_["std_test_score"][best],
                **{f"fold_{fold}_f1": search.cv_results_[f"split{fold}_test_score"][best]
                   for fold in range(5)},
            })
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(destination, index=False, float_format="%.6f")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "outputs/experiment")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/texture_comparison.csv")
    args = parser.parse_args()
    for result in compare(args.experiment, args.output):
        print(f"{result['descriptor']} + {result['classifier']}: "
              f"F1={result['mean_f1']:.4f} ± {result['std_f1']:.4f}")


if __name__ == "__main__":
    main()
