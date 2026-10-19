"""Experimento em etapas explícitas: prepare, extract, select e evaluate."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy
import sklearn
import skimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.download_data import matches
from scripts.image_processing import filtro_bilateral, leaf_features, segment_leaf

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
METHODS = ("otsu", "kmeans", "unmasked")
REPRESENTATIONS = (
    ("kmeans", "hsv", "color_texture"),
    ("otsu", "hsv", "color_texture"),
    ("unmasked", "hsv", "color_texture"),
    ("kmeans", "rgb", "color_texture"),
    ("kmeans", "hsv", "color"),
    ("kmeans", "hsv", "texture"),
    ("kmeans", "hsv", "shape"),
    ("kmeans", "hsv", "all"),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_image(relative):
    image = cv2.imread(str(ROOT / "data/raw" / relative), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {relative}")
    return image


def source_hashes():
    return {name: digest(ROOT / name) for name in
            ("scripts/image_processing.py", "scripts/run_experiments.py", "requirements.txt")}


def duplicate_groups(frame, thumbnails):
    """Agrupa cópias exatas e imagens quase idênticas, sem usar a classe."""
    parent = list(range(len(frame)))
    pairs = []

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    hashes = [int(value, 16) for value in frame.dhash]
    pixels = frame.pixel_sha256.tolist()
    labels = frame["class"].tolist()
    for right in range(len(frame)):
        for left in range(right):
            exact = pixels[left] == pixels[right]
            similar = (hashes[left] ^ hashes[right]).bit_count() <= 4
            if not exact and not similar:
                continue
            if not exact and np.abs(thumbnails[left] - thumbnails[right]).mean() > 8:
                continue
            if exact and labels[left] != labels[right]:
                raise ValueError("Identical decoded images have conflicting labels.")
            parent[find(right)] = find(left)
            pairs.append({"left": frame.iloc[left].path, "right": frame.iloc[right].path,
                          "kind": "exact" if exact else "perceptual"})
    return [f"g{find(i):04d}" for i in range(len(frame))], pairs


def assign_partitions(frame):
    frame = frame.copy()
    # Os seis primeiros arquivos/classe participaram do experimento histórico.
    exposed = frame.groupby("class", sort=True).head(6).group.unique()
    frame["split"] = "calibration"
    frame["fold"] = -1
    eligible = frame.loc[~frame.group.isin(exposed)]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(outer.split(eligible, eligible["class"], eligible.group))
    frame.loc[eligible.index[train], "split"] = "development"
    frame.loc[eligible.index[test], "split"] = "test"
    development = frame.loc[frame.split == "development"]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fold, (_, valid) in enumerate(inner.split(development, development["class"], development.group)):
        frame.loc[development.index[valid], "fold"] = fold
    if frame.groupby("group").split.nunique().max() != 1:
        raise ValueError("A duplicate group crossed partitions.")
    for _, part in frame.loc[frame.split == "development"].groupby("fold"):
        if set(part["class"]) != set(frame["class"]):
            raise ValueError("A validation fold is missing a class.")
    if development.groupby("group").size().empty:
        raise ValueError("No development samples.")
    return frame


def prepare(output):
    if (output / "protocol.json").exists():
        raise FileExistsError("Protocol already exists; use a new output directory for a new experiment.")
    manifest = json.loads((ROOT / "data/download_manifest.json").read_text())
    entries = sorted((e for e in manifest["files"] if e["path"].startswith("without/")), key=lambda e: e["path"])
    records, thumbnails = [], []
    for entry in entries:
        if not matches(ROOT / "data/raw" / entry["path"], entry):
            raise ValueError(f"Missing or invalid image: {entry['path']}")
        image = read_image(entry["path"])
        gray = cv2.cvtColor(cv2.resize(image, (9, 8)), cv2.COLOR_BGR2GRAY)
        bits = (gray[:, 1:] > gray[:, :-1]).ravel()
        dhash = int.from_bytes(np.packbits(bits).tobytes(), "big")
        records.append({"path": entry["path"], "class": Path(entry["path"]).parent.name,
                        "sha256": entry["sha256"], "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "dhash": f"{dhash:016x}", "height": image.shape[0], "width": image.shape[1]})
        thumbnails.append(cv2.resize(image, (32, 32)).astype(np.int16))
    frame = pd.DataFrame(records)
    frame["group"], pairs = duplicate_groups(frame, thumbnails)
    frame = assign_partitions(frame)
    frame.to_csv(output / "partitions.csv", index=False)
    pd.DataFrame(pairs, columns=["left", "right", "kind"]).to_csv(output / "duplicate_pairs.csv", index=False)
    protocol = {
        "seed": SEED, "manifest_sha256": digest(ROOT / "data/download_manifest.json"),
        "partitions_sha256": digest(output / "partitions.csv"), "source_hashes": source_hashes(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__, "skimage": skimage.__version__,
                     "sklearn": sklearn.__version__},
        "filter": {"method": "bilateral", "diameter": 11, "sigma_color": 30, "sigma_space": 5},
        "segmentation": {"otsu": "HSV saturation", "kmeans": "Lab a/b, k=3, stride=4, attempts=3, seed=42",
                         "closing_kernel": "5x5 ellipse", "valid_area_fraction": [.01, .98]},
        "texture": {"levels": 32, "distances": [1, 2, 4], "angles_degrees": [0, 45, 90, 135],
                    "lbp": "uniform, P=8, R=1, interior neighborhoods only"},
        "color_bins": 16, "representations": REPRESENTATIONS,
        "failed_masks": "NaN descriptors; training-only median imputation; all images scored",
        "calibration": "First six lexicographically sorted files per class and their duplicate groups",
        "grouping_limit": "dHash<=4 and thumbnail MAE<=8; no biological individual identifiers",
    }
    save_json(output / "protocol.json", protocol)
    print(frame.groupby(["split", "class"]).size().to_string(), flush=True)
    print(f"Duplicate pairs: {len(pairs)}; groups: {frame.group.nunique()}", flush=True)


def read_partitions(output):
    return pd.read_csv(output / "partitions.csv", dtype={"dhash": str, "sha256": str, "pixel_sha256": str})


def check_protocol(output):
    protocol = json.loads((output / "protocol.json").read_text())
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Processing code changed after protocol registration. Use a new experiment directory.")
    if protocol["partitions_sha256"] != digest(output / "partitions.csv"):
        raise ValueError("Partitions changed after registration.")
    return read_partitions(output)


def extract(output):
    frame = check_protocol(output)
    if (output / "features.csv.gz").exists():
        raise FileExistsError("Features already exist; refusing silent replacement.")
    template = leaf_features(np.zeros((16, 16, 3), np.uint8), np.ones((16, 16), bool))
    records, diagnostics = [], []
    for i, row in frame.iterrows():
        if digest(ROOT / "data/raw" / row.path) != row.sha256:
            raise ValueError(f"Image changed after audit: {row.path}")
        image = filtro_bilateral(read_image(row.path))
        for method in METHODS:
            error, fraction = "", np.nan
            try:
                mask = np.ones(image.shape[:2], bool) if method == "unmasked" else segment_leaf(image, method)
                fraction = float(mask.mean())
                features = leaf_features(image, mask)
                if not np.isfinite(list(features.values())).all():
                    raise ValueError("Nonfinite descriptors.")
            except ValueError as exc:
                error = str(exc)
                features = dict.fromkeys(template, np.nan)
            records.append({"path": row.path, "method": method, **features})
            diagnostics.append({"path": row.path, "method": method, "mask_fraction": fraction, "error": error})
        if (i + 1) % 100 == 0:
            print(f"Extracted {i+1}/{len(frame)}", flush=True)
    pd.DataFrame(records).to_csv(output / "features.csv.gz", index=False, float_format="%.10g")
    pd.DataFrame(diagnostics).to_csv(output / "segmentation.csv", index=False)
    save_json(output / "extraction.json", {"features_sha256": digest(output / "features.csv.gz"),
                                          "columns": list(template), "rows": len(records)})


def feature_columns(frame, space, family):
    prefixes = {"color": (space + "_",), "texture": ("texture_",), "shape": ("shape_",),
                "color_texture": (space + "_", "texture_"),
                "all": (space + "_", "texture_", "shape_")}
    return [column for column in frame if column.startswith(prefixes[family])]


def model_grid(name):
    if name == "svm":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             SVC(class_weight="balanced", kernel="rbf")), {"svc__C": [1, 10]}
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                                    random_state=SEED, n_jobs=1)), {"randomforestclassifier__min_samples_leaf": [1, 3]}
    raise ValueError(f"Unknown classifier: {name}")


def load_features(output):
    partitions = check_protocol(output)
    extraction = json.loads((output / "extraction.json").read_text())
    if extraction["features_sha256"] != digest(output / "features.csv.gz"):
        raise ValueError("Feature file changed after extraction.")
    features = pd.read_csv(output / "features.csv.gz")
    if features.duplicated(["path", "method"]).any() or len(features) != len(partitions)*len(METHODS):
        raise ValueError("Missing or duplicate feature rows.")
    return features.merge(partitions, on="path", validate="many_to_one")


def select(output):
    if (output / "frozen.json").exists():
        raise FileExistsError("Selection is already frozen.")
    frame = load_features(output)
    frame = frame.loc[frame.split == "development"]
    results, frozen = [], []
    for method, space, family in REPRESENTATIONS:
        subset = frame.loc[frame.method == method].reset_index(drop=True)
        columns = feature_columns(subset, space, family)
        cv = [(np.flatnonzero(subset.fold != fold), np.flatnonzero(subset.fold == fold)) for fold in range(5)]
        for train, valid in cv:
            if set(subset.iloc[train].group) & set(subset.iloc[valid].group):
                raise ValueError("A duplicate group crossed validation folds.")
        for classifier in ("svm", "rf"):
            experiment = f"{method}_{space}_{family}_{classifier}"
            estimator, grid = model_grid(classifier)
            search = GridSearchCV(estimator, grid, cv=cv, scoring="f1_macro",
                                  n_jobs=1, refit=False, error_score="raise")
            search.fit(subset[columns], subset["class"])
            for candidate, params in enumerate(search.cv_results_["params"]):
                results.append({"experiment": experiment, "params": json.dumps(params, sort_keys=True),
                                "mean_f1": search.cv_results_["mean_test_score"][candidate],
                                "std_f1": search.cv_results_["std_test_score"][candidate],
                                **{f"fold_{fold}_f1": search.cv_results_[f"split{fold}_test_score"][candidate]
                                   for fold in range(5)}})
            frozen.append({"experiment": experiment, "method": method, "space": space,
                           "family": family, "classifier": classifier, "columns": columns,
                           "params": search.best_params_, "validation_f1": search.best_score_})
            pd.DataFrame(results).to_csv(output / "validation.csv", index=False)
            print(f"{experiment}: validation F1={search.best_score_:.4f}", flush=True)
    winner = max(frozen, key=lambda result: result["validation_f1"])["experiment"]
    save_json(output / "frozen.json", {"winner": winner, "experiments": frozen,
                                      "protocol_sha256": digest(output / "protocol.json"),
                                      "features_sha256": digest(output / "features.csv.gz")})
    print(f"Frozen winner: {winner}. Test has not been evaluated.", flush=True)


def classification_metrics(truth, predictions, classes):
    matrix = confusion_matrix(truth, predictions, labels=classes)
    per_class = []
    for index, label in enumerate(classes):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_class.append({"class": label, "support": int(tp+fn), "sensitivity": float(tp/(tp+fn)),
                          "specificity": float(tn/(tn+fp))})
    return float(f1_score(truth, predictions, labels=classes, average="macro")), matrix, per_class


def evaluate(output):
    if (output / "test_started.json").exists():
        raise FileExistsError("The test was already opened. Do not tune or overwrite; inspect the recorded results.")
    frozen = json.loads((output / "frozen.json").read_text())
    if frozen["protocol_sha256"] != digest(output / "protocol.json"):
        raise ValueError("Protocol changed after model selection.")
    if frozen["features_sha256"] != digest(output / "features.csv.gz"):
        raise ValueError("Features changed after model selection.")
    frame = load_features(output)
    save_json(output / "test_started.json", {"frozen_sha256": digest(output / "frozen.json"), "winner": frozen["winner"]})
    summary, predictions, per_class, matrices = [], [], [], {}
    for setting in frozen["experiments"]:
        subset = frame.loc[frame.method == setting["method"]]
        train = subset.loc[subset.split == "development"]
        test = subset.loc[subset.split == "test"]
        estimator, _ = model_grid(setting["classifier"])
        estimator.set_params(**setting["params"])
        estimator.fit(train[setting["columns"]], train["class"])
        predicted = estimator.predict(test[setting["columns"]])
        classes = sorted(train["class"].unique())
        score, matrix, metrics = classification_metrics(test["class"], predicted, classes)
        name = setting["experiment"]
        summary.append({"experiment": name, "validation_f1": setting["validation_f1"], "test_f1": score,
                        "selected_on_validation": name == frozen["winner"]})
        predictions.extend({"experiment": name, "path": path, "truth": truth, "prediction": pred}
                           for path, truth, pred in zip(test.path, test["class"], predicted))
        per_class.extend({"experiment": name, **metric} for metric in metrics)
        matrices[name] = {"classes": classes, "matrix": matrix.tolist()}
        print(f"{name}: test F1={score:.4f}", flush=True)
    pd.DataFrame(summary).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(per_class).to_csv(output / "per_class.csv", index=False)
    save_json(output / "confusion_matrices.json", matrices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "extract", "select", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    {"prepare": prepare, "extract": extract, "select": select, "evaluate": evaluate}[args.stage](args.output)


if __name__ == "__main__":
    main()
