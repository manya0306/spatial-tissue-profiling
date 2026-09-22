"""Multi-gene spatial expression prediction: leakage-safe grouped evaluation.

Predicts the full 56-gene target panel from frozen ResNet18 H&E patch features,
neighbouring-patch image features and within-sample spatial coordinates.

Evaluation is grouped by sample throughout. A whole sample is either in the
training fold or the test fold, never split across both, so no spot ever has a
spatially adjacent neighbour on the other side of the split. Every fitted
object - feature scaler, PCA basis, target scaler, model weights - is fitted on
training folds only.

No target-derived quantity is ever used as an input. Spatial coordinates are
standardised within each sample, which uses only x/y positions and no
expression values.

Run from the project root:

    python scripts\\run_experiments.py
    python scripts\\run_experiments.py --cv loso        (19 folds, slower)
    python scripts\\run_experiments.py --skip-mlp       (linear models only)
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

IMAGE_FEATURE_PATH = PROJECT_ROOT / "outputs" / "image_features" / "resnet18_features.npy"
IMAGE_METADATA_PATH = PROJECT_ROOT / "outputs" / "image_features" / "resnet18_metadata.csv"
NEIGHBOR_FEATURE_PATH = (
    PROJECT_ROOT / "outputs" / "spatial_features" / "neighbor_image_features.npy"
)
TARGET_PATH = PROJECT_ROOT / "outputs" / "gene_targets" / "gene_expression_targets.csv"
TARGET_CONFIG_PATH = PROJECT_ROOT / "outputs" / "gene_targets" / "target_config.json"

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments"

RANDOM_SEED = 42
N_PCA_COMPONENTS = 128
RIDGE_ALPHAS = [1.0, 10.0, 100.0, 1000.0]

MLP_HIDDEN = 256
MLP_EPOCHS = 60
MLP_PATIENCE = 8
MLP_BATCH = 256
MLP_LR = 1e-3
MLP_VAL_SAMPLES = 3  # training samples held out inside each fold for early stopping


@dataclass
class RunConfig:
    """Everything needed to reproduce a run."""

    cv_strategy: str
    n_folds: int
    random_seed: int
    n_pca_components: int
    ridge_alphas: List[float]
    mlp_hidden: int
    mlp_epochs: int
    mlp_patience: int
    mlp_batch_size: int
    mlp_learning_rate: float
    n_spots: int
    n_genes: int
    n_samples: int
    image_feature_dim: int
    neighbor_feature_dim: int
    python_version: str
    platform: str


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """Load features, coordinates, targets and grouping metadata.

    Returns
    -------
    image_features, neighbor_features, coordinates, targets, metadata, gene_names
    """
    for path in [
        IMAGE_FEATURE_PATH,
        IMAGE_METADATA_PATH,
        NEIGHBOR_FEATURE_PATH,
        TARGET_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                "Run scripts/extract_resnet_features.py, "
                "scripts/build_spatial_features.py and "
                "scripts/build_gene_targets.py first."
            )

    image_features = np.load(IMAGE_FEATURE_PATH).astype(np.float32)
    neighbor_features = np.load(NEIGHBOR_FEATURE_PATH).astype(np.float32)
    metadata = pd.read_csv(IMAGE_METADATA_PATH)
    targets_frame = pd.read_csv(TARGET_PATH)

    metadata["barcode"] = metadata["barcode"].astype(str)
    targets_frame["barcode"] = targets_frame["barcode"].astype(str)

    if len(targets_frame) != len(metadata):
        raise ValueError(
            f"Target rows ({len(targets_frame)}) != metadata rows ({len(metadata)})"
        )

    order_matches = bool(
        (targets_frame["sample"].to_numpy() == metadata["sample"].to_numpy()).all()
        and (targets_frame["barcode"].to_numpy() == metadata["barcode"].to_numpy()).all()
    )
    if not order_matches:
        raise ValueError(
            "Target row order does not match resnet18_metadata.csv. "
            "Re-run scripts/build_gene_targets.py."
        )

    if TARGET_CONFIG_PATH.exists():
        with open(TARGET_CONFIG_PATH, "r", encoding="utf-8") as handle:
            gene_names = json.load(handle)["genes"]
    else:
        reserved = {"sample", "barcode", "x_coordinate", "y_coordinate", "library_size"}
        gene_names = [c for c in targets_frame.columns if c not in reserved]

    targets = targets_frame[gene_names].to_numpy(dtype=np.float32)

    coordinates = normalise_coordinates_within_sample(metadata)

    if not np.isfinite(image_features).all():
        raise ValueError("Non-finite values in image features")
    if not np.isfinite(neighbor_features).all():
        raise ValueError("Non-finite values in neighbour features")
    if not np.isfinite(targets).all():
        raise ValueError("Non-finite values in targets")

    return image_features, neighbor_features, coordinates, targets, metadata, gene_names


def normalise_coordinates_within_sample(metadata: pd.DataFrame) -> np.ndarray:
    """Standardise x/y within each sample.

    Raw pixel coordinates are not comparable between samples - each slide has
    its own origin and tissue placement. Standardising within a sample turns
    them into a relative position within that tissue, which is comparable.
    This uses positions only; no expression value is involved.
    """
    frame = metadata[["sample", "x_coordinate", "y_coordinate"]].copy()

    def standardise(group: pd.DataFrame) -> pd.DataFrame:
        result = group.copy()
        for column in ["x_coordinate", "y_coordinate"]:
            values = result[column].to_numpy(dtype=float)
            spread = values.std()
            result[column] = (values - values.mean()) / (spread if spread > 0 else 1.0)
        return result

    standardised = frame.groupby("sample", group_keys=False, sort=False).apply(standardise)
    return standardised[["x_coordinate", "y_coordinate"]].to_numpy(dtype=np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_correlation(truth: np.ndarray, prediction: np.ndarray, method: str) -> float:
    """Correlation that returns NaN rather than raising on degenerate input."""
    if len(truth) < 3 or np.std(truth) == 0 or np.std(prediction) == 0:
        return float("nan")
    try:
        if method == "pearson":
            value = pearsonr(truth, prediction)[0]
        else:
            value = spearmanr(truth, prediction)[0]
        return float(value)
    except Exception:
        return float("nan")


def gene_metrics(truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R² and correlations for a single gene."""
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    total_variance = float(np.sum((truth - truth.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(error ** 2) / total_variance)
        if total_variance > 0
        else float("nan")
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "pearson": safe_correlation(truth, prediction, "pearson"),
        "spearman": safe_correlation(truth, prediction, "spearman"),
    }


def within_sample_pearson(
    truth: np.ndarray,
    prediction: np.ndarray,
    samples: np.ndarray,
) -> float:
    """Mean Pearson correlation computed separately inside each sample.

    This asks whether the model recovers the spatial *pattern* within a tissue
    section, which is the scientific claim being made. It is not affected by
    constant per-slide offsets, which global R² penalises heavily.
    """
    values = []
    for sample in np.unique(samples):
        mask = samples == sample
        if mask.sum() >= 3:
            values.append(safe_correlation(truth[mask], prediction[mask], "pearson"))
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_predict_mean(
    y_train: np.ndarray,
    n_test: int,
) -> np.ndarray:
    """Training-set per-gene mean, broadcast to the test fold."""
    return np.tile(y_train.mean(axis=0), (n_test, 1))


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alphas: List[float],
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Multi-output ridge with alpha chosen on a held-out slice of the training fold.

    Alpha selection never sees the test fold.
    """
    rng = np.random.default_rng(seed)
    n_train = len(x_train)
    indices = rng.permutation(n_train)
    cut = max(1, int(0.8 * n_train))
    inner_train, inner_val = indices[:cut], indices[cut:]

    scaler = StandardScaler().fit(x_train[inner_train])

    best_alpha, best_score = alphas[0], np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha, random_state=seed)
        model.fit(scaler.transform(x_train[inner_train]), y_train[inner_train])
        predictions = model.predict(scaler.transform(x_train[inner_val]))
        score = float(np.mean(np.abs(predictions - y_train[inner_val])))
        if score < best_score:
            best_alpha, best_score = alpha, score

    final_scaler = StandardScaler().fit(x_train)
    final_model = Ridge(alpha=best_alpha, random_state=seed)
    final_model.fit(final_scaler.transform(x_train), y_train)

    return final_model.predict(final_scaler.transform(x_test)), best_alpha


def fit_predict_pca_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_components: int,
    alphas: List[float],
    seed: int,
) -> Tuple[np.ndarray, float]:
    """PCA then ridge. The PCA basis is fitted on the training fold only."""
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)

    components = min(n_components, x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=components, random_state=seed).fit(x_train_scaled)

    z_train = pca.transform(x_train_scaled)
    z_test = pca.transform(scaler.transform(x_test))

    return fit_predict_ridge(z_train, y_train, z_test, alphas, seed)


def fit_predict_mlp(
    image_train: np.ndarray,
    neighbor_train: np.ndarray,
    coord_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    image_test: np.ndarray,
    neighbor_test: np.ndarray,
    coord_test: np.ndarray,
    seed: int,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Multimodal fusion network.

    Three modality-specific projections (patch appearance, neighbourhood
    appearance, relative position) are concatenated and passed through a shared
    trunk to a multi-output head. Early stopping uses whole training samples
    held out inside the training fold, so the validation signal is also
    sample-grouped and the test fold is untouched.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    unique_samples = np.unique(groups_train)
    rng = np.random.default_rng(seed)
    n_val = min(MLP_VAL_SAMPLES, max(1, len(unique_samples) // 4))
    val_samples = rng.choice(unique_samples, size=n_val, replace=False)
    val_mask = np.isin(groups_train, val_samples)
    train_mask = ~val_mask

    if train_mask.sum() == 0:
        train_mask = np.ones(len(groups_train), dtype=bool)
        val_mask = train_mask

    image_scaler = StandardScaler().fit(image_train[train_mask])
    neighbor_scaler = StandardScaler().fit(neighbor_train[train_mask])
    target_scaler = StandardScaler().fit(y_train[train_mask])

    def to_tensor(array: np.ndarray) -> "torch.Tensor":
        return torch.tensor(np.asarray(array, dtype=np.float32), device=device)

    xi_tr = to_tensor(image_scaler.transform(image_train[train_mask]))
    xn_tr = to_tensor(neighbor_scaler.transform(neighbor_train[train_mask]))
    xc_tr = to_tensor(coord_train[train_mask])
    y_tr = to_tensor(target_scaler.transform(y_train[train_mask]))

    xi_va = to_tensor(image_scaler.transform(image_train[val_mask]))
    xn_va = to_tensor(neighbor_scaler.transform(neighbor_train[val_mask]))
    xc_va = to_tensor(coord_train[val_mask])
    y_va = to_tensor(target_scaler.transform(y_train[val_mask]))

    class FusionNet(nn.Module):
        def __init__(self, image_dim: int, neighbor_dim: int, coord_dim: int, n_out: int):
            super().__init__()
            self.image_projection = nn.Sequential(
                nn.Linear(image_dim, MLP_HIDDEN), nn.ReLU(), nn.Dropout(0.2)
            )
            self.neighbor_projection = nn.Sequential(
                nn.Linear(neighbor_dim, MLP_HIDDEN // 2), nn.ReLU(), nn.Dropout(0.2)
            )
            self.coord_embedding = nn.Sequential(
                nn.Linear(coord_dim, 32), nn.ReLU()
            )
            fused_dim = MLP_HIDDEN + MLP_HIDDEN // 2 + 32
            self.trunk = nn.Sequential(
                nn.Linear(fused_dim, MLP_HIDDEN),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(MLP_HIDDEN, n_out),
            )

        def forward(self, image, neighbor, coords):
            fused = torch.cat(
                [
                    self.image_projection(image),
                    self.neighbor_projection(neighbor),
                    self.coord_embedding(coords),
                ],
                dim=1,
            )
            return self.trunk(fused)

    model = FusionNet(
        image_train.shape[1], neighbor_train.shape[1], coord_train.shape[1], y_train.shape[1]
    ).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=MLP_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    n = len(xi_tr)
    best_val, best_state, best_epoch, patience = np.inf, None, 0, 0
    history: List[Dict[str, float]] = []

    for epoch in range(MLP_EPOCHS):
        model.train()
        permutation = torch.randperm(n)
        epoch_loss = 0.0

        for start in range(0, n, MLP_BATCH):
            batch = permutation[start : start + MLP_BATCH]
            optimiser.zero_grad()
            output = model(xi_tr[batch], xn_tr[batch], xc_tr[batch])
            loss = criterion(output, y_tr[batch])
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.item()) * len(batch)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xi_va, xn_va, xc_va), y_va).item())

        scheduler.step(val_loss)
        history.append(
            {"epoch": epoch + 1, "train_loss": epoch_loss / n, "val_loss": val_loss}
        )

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, patience = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= MLP_PATIENCE:
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}  train={epoch_loss/n:.4f}  val={val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        scaled_predictions = (
            model(
                to_tensor(image_scaler.transform(image_test)),
                to_tensor(neighbor_scaler.transform(neighbor_test)),
                to_tensor(coord_test),
            )
            .cpu()
            .numpy()
        )

    predictions = target_scaler.inverse_transform(scaled_predictions)

    info = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "epochs_run": len(history),
        "validation_samples": [str(s) for s in val_samples],
        "history": history,
    }
    return predictions, info


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    start_time = time.time()
    np.random.seed(RANDOM_SEED)

    print("\nMULTI-GENE SPATIAL EXPRESSION PREDICTION")
    print("=" * 78)

    image, neighbor, coords, targets, metadata, genes = load_dataset()
    groups = metadata["sample"].to_numpy()

    print(f"Spots            : {len(image)}")
    print(f"Genes            : {len(genes)}")
    print(f"Samples          : {len(np.unique(groups))}")
    print(f"Image features   : {image.shape}")
    print(f"Neighbour feats  : {neighbor.shape}")
    print(f"Coordinates      : {coords.shape} (standardised within sample)")

    if args.cv == "loso":
        splitter = LeaveOneGroupOut()
        n_folds = len(np.unique(groups))
    else:
        n_folds = min(args.folds, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=n_folds)

    print(f"CV strategy      : {args.cv} ({n_folds} folds, grouped by sample)")

    image_plus_neighbor = np.hstack([image, neighbor])
    image_neighbor_coords = np.hstack([image, neighbor, coords])

    model_names = [
        "mean_baseline",
        "ridge_image",
        "ridge_image_neighbor",
        "pca_ridge_image",
        "pca_ridge_image_neighbor_coords",
    ]
    if not args.skip_mlp:
        model_names.append("mlp_fusion")

    # Out-of-fold prediction store: every spot is predicted exactly once, by a
    # model that never saw its sample during training.
    predictions: Dict[str, np.ndarray] = {
        name: np.full_like(targets, np.nan) for name in model_names
    }
    fold_assignment = np.full(len(targets), -1, dtype=int)
    fold_records: List[Dict[str, object]] = []
    mlp_histories: List[Dict[str, object]] = []

    for fold, (train_index, test_index) in enumerate(
        splitter.split(image, targets, groups), start=1
    ):
        test_samples = sorted(np.unique(groups[test_index]))
        fold_assignment[test_index] = fold

        print(
            f"\nFold {fold}/{n_folds}  train={len(train_index):6d}  "
            f"test={len(test_index):6d}  held-out samples: {', '.join(test_samples)}"
        )

        y_train, y_test = targets[train_index], targets[test_index]

        # --- mean baseline -------------------------------------------------
        predictions["mean_baseline"][test_index] = fit_predict_mean(
            y_train, len(test_index)
        )
        print("   mean_baseline                   done")

        # --- ridge on image features ---------------------------------------
        preds, alpha = fit_predict_ridge(
            image[train_index], y_train, image[test_index], RIDGE_ALPHAS, RANDOM_SEED
        )
        predictions["ridge_image"][test_index] = preds
        print(f"   ridge_image                     done (alpha={alpha:g})")

        # --- ridge on image + neighbour features ---------------------------
        preds, alpha = fit_predict_ridge(
            image_plus_neighbor[train_index],
            y_train,
            image_plus_neighbor[test_index],
            RIDGE_ALPHAS,
            RANDOM_SEED,
        )
        predictions["ridge_image_neighbor"][test_index] = preds
        print(f"   ridge_image_neighbor            done (alpha={alpha:g})")

        # --- PCA + ridge, image only ---------------------------------------
        preds, alpha = fit_predict_pca_ridge(
            image[train_index],
            y_train,
            image[test_index],
            N_PCA_COMPONENTS,
            RIDGE_ALPHAS,
            RANDOM_SEED,
        )
        predictions["pca_ridge_image"][test_index] = preds
        print(f"   pca_ridge_image                 done (alpha={alpha:g})")

        # --- PCA + ridge, image + neighbour + coordinates ------------------
        preds, alpha = fit_predict_pca_ridge(
            image_neighbor_coords[train_index],
            y_train,
            image_neighbor_coords[test_index],
            N_PCA_COMPONENTS,
            RIDGE_ALPHAS,
            RANDOM_SEED,
        )
        predictions["pca_ridge_image_neighbor_coords"][test_index] = preds
        print(f"   pca_ridge_image_neighbor_coords done (alpha={alpha:g})")

        # --- multimodal fusion network -------------------------------------
        if not args.skip_mlp:
            try:
                preds, info = fit_predict_mlp(
                    image[train_index],
                    neighbor[train_index],
                    coords[train_index],
                    y_train,
                    groups[train_index],
                    image[test_index],
                    neighbor[test_index],
                    coords[test_index],
                    RANDOM_SEED,
                    verbose=args.verbose,
                )
                predictions["mlp_fusion"][test_index] = preds
                info["fold"] = fold
                mlp_histories.append(info)
                print(
                    f"   mlp_fusion                      done "
                    f"(best epoch {info['best_epoch']}/{info['epochs_run']}, "
                    f"val={info['best_val_loss']:.4f})"
                )
            except ImportError:
                print("   mlp_fusion                      SKIPPED (PyTorch not installed)")
                if "mlp_fusion" in model_names:
                    model_names.remove("mlp_fusion")
                    predictions.pop("mlp_fusion", None)

        fold_records.append(
            {
                "fold": fold,
                "n_train": int(len(train_index)),
                "n_test": int(len(test_index)),
                "test_samples": ", ".join(test_samples),
            }
        )

    # ----------------------------------------------------------- metrics ---

    print("\n" + "=" * 78)
    print("COMPUTING METRICS")
    print("=" * 78)

    per_gene_rows: List[Dict[str, object]] = []
    per_fold_rows: List[Dict[str, object]] = []

    for model_name in model_names:
        prediction_matrix = predictions[model_name]
        if np.isnan(prediction_matrix).all():
            continue

        for gene_index, gene in enumerate(genes):
            truth = targets[:, gene_index]
            predicted = prediction_matrix[:, gene_index]
            valid = ~np.isnan(predicted)

            metrics = gene_metrics(truth[valid], predicted[valid])
            metrics.update(
                {
                    "model": model_name,
                    "gene": gene,
                    "within_sample_pearson": within_sample_pearson(
                        truth[valid], predicted[valid], groups[valid]
                    ),
                    "n_spots": int(valid.sum()),
                }
            )
            per_gene_rows.append(metrics)

        for fold in sorted(set(fold_assignment[fold_assignment > 0])):
            mask = fold_assignment == fold
            for gene_index, gene in enumerate(genes):
                truth = targets[mask, gene_index]
                predicted = prediction_matrix[mask, gene_index]
                if np.isnan(predicted).any():
                    continue
                metrics = gene_metrics(truth, predicted)
                metrics.update({"model": model_name, "gene": gene, "fold": int(fold)})
                per_fold_rows.append(metrics)

    per_gene = pd.DataFrame(per_gene_rows)
    per_fold = pd.DataFrame(per_fold_rows)

    fold_spread = (
        per_fold.groupby(["model", "gene"])
        .agg(
            MAE_fold_mean=("MAE", "mean"),
            MAE_fold_std=("MAE", "std"),
            R2_fold_mean=("R2", "mean"),
            R2_fold_std=("R2", "std"),
        )
        .reset_index()
    )
    per_gene = per_gene.merge(fold_spread, on=["model", "gene"], how="left")

    aggregate = (
        per_gene.groupby("model")
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            R2_median=("R2", "median"),
            genes_with_positive_R2=("R2", lambda s: int((s > 0).sum())),
            pearson_mean=("pearson", "mean"),
            spearman_mean=("spearman", "mean"),
            within_sample_pearson_mean=("within_sample_pearson", "mean"),
            n_genes=("gene", "count"),
        )
        .reset_index()
        .sort_values("within_sample_pearson_mean", ascending=False)
    )

    # ------------------------------------------------------------ saving ---

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "predictions").mkdir(exist_ok=True)

    per_gene.to_csv(OUTPUT_ROOT / "metrics_per_gene.csv", index=False)
    per_fold.to_csv(OUTPUT_ROOT / "metrics_per_gene_per_fold.csv", index=False)
    aggregate.to_csv(OUTPUT_ROOT / "metrics_aggregate.csv", index=False)

    assignment_frame = metadata[["sample", "barcode", "x_coordinate", "y_coordinate"]].copy()
    assignment_frame["fold"] = fold_assignment
    assignment_frame.to_csv(OUTPUT_ROOT / "fold_assignments.csv", index=False)
    pd.DataFrame(fold_records).to_csv(OUTPUT_ROOT / "fold_summary.csv", index=False)

    truth_frame = assignment_frame.copy()
    for gene_index, gene in enumerate(genes):
        truth_frame[gene] = targets[:, gene_index]
    truth_frame.to_csv(OUTPUT_ROOT / "predictions" / "ground_truth.csv", index=False)

    for model_name in model_names:
        if model_name not in predictions:
            continue
        frame = assignment_frame.copy()
        for gene_index, gene in enumerate(genes):
            frame[gene] = predictions[model_name][:, gene_index]
        frame.to_csv(OUTPUT_ROOT / "predictions" / f"{model_name}.csv", index=False)

    if mlp_histories:
        with open(OUTPUT_ROOT / "mlp_training_history.json", "w", encoding="utf-8") as handle:
            json.dump(mlp_histories, handle, indent=2)

    config = RunConfig(
        cv_strategy=args.cv,
        n_folds=n_folds,
        random_seed=RANDOM_SEED,
        n_pca_components=N_PCA_COMPONENTS,
        ridge_alphas=RIDGE_ALPHAS,
        mlp_hidden=MLP_HIDDEN,
        mlp_epochs=MLP_EPOCHS,
        mlp_patience=MLP_PATIENCE,
        mlp_batch_size=MLP_BATCH,
        mlp_learning_rate=MLP_LR,
        n_spots=int(len(targets)),
        n_genes=len(genes),
        n_samples=int(len(np.unique(groups))),
        image_feature_dim=int(image.shape[1]),
        neighbor_feature_dim=int(neighbor.shape[1]),
        python_version=platform.python_version(),
        platform=platform.platform(),
    )
    with open(OUTPUT_ROOT / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "models": model_names,
                "genes": genes,
                "runtime_seconds": round(time.time() - start_time, 1),
            },
            handle,
            indent=2,
        )

    write_results_markdown(aggregate, per_gene, config, model_names, genes)

    # ----------------------------------------------------------- display ---

    print("\n" + "=" * 78)
    print("AGGREGATE RESULTS (mean over all genes, out-of-fold)")
    print("=" * 78)
    print(
        aggregate[
            [
                "model",
                "MAE",
                "RMSE",
                "R2_mean",
                "genes_with_positive_R2",
                "pearson_mean",
                "within_sample_pearson_mean",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    best_model = aggregate.iloc[0]["model"]
    print(f"\nBest by mean within-sample Pearson: {best_model}")

    top = (
        per_gene[per_gene["model"] == best_model]
        .sort_values("within_sample_pearson", ascending=False)
        .head(15)
    )
    print(f"\nTOP 15 GENES for {best_model}")
    print(
        top[["gene", "MAE", "R2", "pearson", "within_sample_pearson"]]
        .round(4)
        .to_string(index=False)
    )

    worst = (
        per_gene[per_gene["model"] == best_model]
        .sort_values("within_sample_pearson")
        .head(5)
    )
    print(f"\nWORST 5 GENES for {best_model}")
    print(
        worst[["gene", "MAE", "R2", "pearson", "within_sample_pearson"]]
        .round(4)
        .to_string(index=False)
    )

    print(f"\nRuntime: {time.time() - start_time:.1f}s")
    print(f"Saved to: {OUTPUT_ROOT.relative_to(PROJECT_ROOT)}")
    print("\nNext: python scripts\\make_figures.py")
    return 0


def write_results_markdown(
    aggregate: pd.DataFrame,
    per_gene: pd.DataFrame,
    config: RunConfig,
    model_names: List[str],
    genes: List[str],
) -> None:
    """Write RESULTS.md directly from the computed metrics.

    Generating this file from the metric tables rather than by hand is a
    deliberate integrity measure: no number in it can drift from what was
    actually computed.
    """
    best_model = aggregate.iloc[0]["model"]
    best_rows = per_gene[per_gene["model"] == best_model].sort_values(
        "within_sample_pearson", ascending=False
    )

    lines: List[str] = []
    lines.append("# Results\n")
    lines.append(
        "All numbers on this page were generated by `scripts/run_experiments.py` "
        "and written automatically. They are out-of-fold predictions under "
        "sample-grouped cross-validation: every spot was predicted by a model "
        "that never saw its tissue sample during training.\n"
    )
    lines.append("## Run configuration\n")
    lines.append(f"- Spots: {config.n_spots}")
    lines.append(f"- Genes: {config.n_genes}")
    lines.append(f"- Samples: {config.n_samples}")
    lines.append(f"- CV strategy: {config.cv_strategy}, {config.n_folds} folds, grouped by sample")
    lines.append(f"- Random seed: {config.random_seed}")
    lines.append(f"- Image features: frozen ImageNet ResNet18, dim {config.image_feature_dim}")
    lines.append(f"- Neighbour features: mean of 6 nearest patches, dim {config.neighbor_feature_dim}\n")

    lines.append("## Model comparison\n")
    lines.append(
        "| Model | MAE | RMSE | Mean R2 | Genes with R2>0 | Mean Pearson | Mean within-sample Pearson |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in aggregate.iterrows():
        lines.append(
            f"| {row['model']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | "
            f"{row['R2_mean']:.4f} | {int(row['genes_with_positive_R2'])}/{int(row['n_genes'])} | "
            f"{row['pearson_mean']:.4f} | {row['within_sample_pearson_mean']:.4f} |"
        )
    lines.append("")

    lines.append(f"## Per-gene results, best model (`{best_model}`)\n")
    lines.append("Ordered by within-sample Pearson. All 56 genes are shown; none were dropped.\n")
    lines.append("| Gene | MAE | RMSE | R2 | Pearson | Spearman | Within-sample Pearson |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in best_rows.iterrows():
        lines.append(
            f"| {row['gene']} | {row['MAE']:.4f} | {row['RMSE']:.4f} | {row['R2']:.4f} | "
            f"{row['pearson']:.4f} | {row['spearman']:.4f} | {row['within_sample_pearson']:.4f} |"
        )
    lines.append("")

    positive = int((best_rows["R2"] > 0).sum())
    lines.append("## Summary\n")
    lines.append(
        f"- {positive} of {len(genes)} genes reached positive R2 under the best model "
        f"(`{best_model}`)."
    )
    lines.append(
        f"- Mean within-sample Pearson correlation across all genes: "
        f"{best_rows['within_sample_pearson'].mean():.4f}."
    )
    lines.append(
        "- Genes with negative R2 are reported above unchanged. A negative R2 means the "
        "model predicted that gene worse than the training-set mean would have.\n"
    )
    lines.append("## Limitations\n")
    lines.append(
        "- 19 tissue sections from a single public dataset (GSE197023). Sample-grouped "
        "CV controls leakage but cannot substitute for an independent cohort."
    )
    lines.append(
        "- Image features come from a frozen ImageNet-pretrained ResNet18, not a "
        "pathology-specific encoder. No image backbone was fine-tuned."
    )
    lines.append(
        "- Targets are log-normalised counts for 56 genes, not a full transcriptome."
    )
    lines.append(
        "- Visium spots are roughly 55 microns and contain several cells; predictions "
        "are spot-level, not cell-level."
    )
    lines.append(
        "- This is an educational research-inspired prototype. It is not a clinical or "
        "diagnostic system, and it is not a reproduction of OmiCLIP or STPath.\n"
    )

    with open(PROJECT_ROOT / "RESULTS.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cv",
        choices=["groupkfold", "loso"],
        default="groupkfold",
        help="Cross-validation strategy (both are grouped by sample)",
    )
    parser.add_argument("--folds", type=int, default=5, help="Folds for groupkfold")
    parser.add_argument("--skip-mlp", action="store_true", help="Linear models only")
    parser.add_argument("--verbose", action="store_true", help="Print training progress")
    args = parser.parse_args()

    try:
        return run(args)
    except FileNotFoundError as error:
        print(f"\nERROR: {error}")
        return 1
    except ValueError as error:
        print(f"\nERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
