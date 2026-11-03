#!/usr/bin/env python
"""Minimal reproduction code for edge-private post-release link prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as ssp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score


Edge = tuple[int, int]


def canonical(edge: Edge) -> Edge:
    return (min(edge), max(edge))


def graph_edges(graph: ssp.csr_matrix) -> set[Edge]:
    upper = ssp.triu(graph, k=1).tocoo()
    return {
        (int(u), int(v))
        for u, v in zip(upper.row.tolist(), upper.col.tolist())
    }


def load_graph_data(path: Path) -> tuple[ssp.csr_matrix, ssp.csr_matrix]:
    mat = sio.loadmat(path)
    graph = ssp.csr_matrix(mat["net"], dtype=np.int8)
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1
    features = ssp.csr_matrix(mat["group"], dtype=np.float32)
    return graph, features


def split_bucket(edge: Edge, salt: int) -> int:
    left, right = canonical(edge)
    payload = f"{left}:{right}:{salt}".encode("ascii")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(),
        "little",
    ) % 100


def split_name(bucket: int) -> str:
    if bucket < 5:
        return "valid"
    if bucket < 15:
        return "test"
    return "train"


def feature_candidates(
    features: ssp.csr_matrix,
    candidates_per_node: int,
    block_size: int = 256,
) -> list[Edge]:
    row_norm = np.sqrt(
        np.asarray(features.multiply(features).sum(axis=1)).ravel()
    )
    inverse = np.zeros_like(row_norm)
    nonzero = row_norm > 0
    inverse[nonzero] = 1.0 / row_norm[nonzero]
    normalized = ssp.diags(inverse) @ features
    take = min(candidates_per_node, features.shape[0] - 1)
    pairs: set[Edge] = set()
    for start in range(0, features.shape[0], block_size):
        end = min(start + block_size, features.shape[0])
        similarity = np.asarray(
            (normalized[start:end] @ normalized.T).todense(),
            dtype=np.float32,
        )
        rows = np.arange(end - start)
        similarity[rows, rows + start] = -np.inf
        top = np.argpartition(-similarity, kth=take - 1, axis=1)[:, :take]
        for offset, row in enumerate(top):
            source = start + offset
            pairs.update(
                (min(source, int(target)), max(source, int(target)))
                for target in row
            )
    return sorted(pairs)


def public_projected_features(
    features: ssp.csr_matrix,
    dimension: int = 256,
) -> np.ndarray:
    rng = np.random.default_rng(20261031)
    projection = rng.choice(
        np.asarray([-1.0, 1.0], dtype=np.float32),
        size=(features.shape[1], dimension),
    ) / math.sqrt(dimension)
    projected = np.asarray(features @ projection, dtype=np.float32)
    projected /= np.maximum(
        np.linalg.norm(projected, axis=1, keepdims=True),
        1e-12,
    )
    return projected


def sparse_pair_dot(
    features: ssp.csr_matrix,
    pairs: list[Edge],
    chunk_size: int = 2000,
) -> np.ndarray:
    output = np.empty(len(pairs), dtype=np.float32)
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start : start + chunk_size]
        rows_u = np.asarray([u for u, _ in chunk], dtype=np.int32)
        rows_v = np.asarray([v for _, v in chunk], dtype=np.int32)
        output[start : start + len(chunk)] = np.asarray(
            features[rows_u].multiply(features[rows_v]).sum(axis=1).ravel(),
            dtype=np.float32,
        )
    return output


def pair_features_for(
    raw_features: ssp.csr_matrix,
    projected: np.ndarray,
    norms: np.ndarray,
    pairs: list[Edge],
) -> np.ndarray:
    dots = sparse_pair_dot(raw_features, pairs)
    u = np.asarray([edge[0] for edge in pairs], dtype=np.int32)
    v = np.asarray([edge[1] for edge in pairs], dtype=np.int32)
    cosine = dots / np.maximum(norms[u] * norms[v], 1e-12)
    product = projected[u] * projected[v]
    scalars = np.column_stack(
        [
            cosine,
            np.log1p(dots) / 10.0,
            np.abs(
                np.log1p(np.linalg.norm(projected[u], axis=1))
                - np.log1p(np.linalg.norm(projected[v], axis=1))
            )
            / 5.0,
        ]
    ).astype(np.float32)
    return np.concatenate([product, scalars], axis=1)


def randomized_response_graph(
    candidate_edges: list[Edge],
    true_edges: set[Edge],
    epsilon: float,
    seed: int,
) -> set[Edge]:
    rng = np.random.default_rng(seed)
    present = np.fromiter(
        (edge in true_edges for edge in candidate_edges),
        dtype=bool,
        count=len(candidate_edges),
    )
    probability = np.where(
        present,
        math.exp(epsilon) / (1.0 + math.exp(epsilon)),
        1.0 / (1.0 + math.exp(epsilon)),
    )
    released = rng.random(len(candidate_edges)) < probability
    return {
        edge
        for edge, keep in zip(candidate_edges, released)
        if keep
    }


def jaccard_denoise(
    candidate_edges: list[Edge],
    released_edges: set[Edge],
    node_count: int,
    target_average_degree: float,
    epsilon: float,
    seed: int,
) -> set[Edge]:
    rows = np.asarray([u for u, _ in released_edges], dtype=np.int32)
    cols = np.asarray([v for _, v in released_edges], dtype=np.int32)
    adjacency = ssp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(node_count, node_count),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    common = (adjacency @ adjacency.T).tocsr()
    candidate_rows = np.asarray(
        [u for u, _ in candidate_edges],
        dtype=np.int32,
    )
    candidate_cols = np.asarray(
        [v for _, v in candidate_edges],
        dtype=np.int32,
    )
    common_count = np.asarray(
        common[candidate_rows, candidate_cols]
    ).ravel()
    union_count = (
        degree[candidate_rows]
        + degree[candidate_cols]
        - common_count
    )
    jaccard = np.divide(
        common_count,
        union_count,
        out=np.zeros_like(common_count, dtype=np.float64),
        where=union_count > 0,
    )
    tie_break = np.random.default_rng(seed).random(len(candidate_edges))
    target_count = int(round(target_average_degree * node_count / 2.0))
    keep_probability = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    real_count = int(round(keep_probability * target_count))
    released_mask = np.asarray(
        [edge in released_edges for edge in candidate_edges],
        dtype=bool,
    )
    real_indices = np.flatnonzero(released_mask)
    fake_indices = np.flatnonzero(~released_mask)
    real_order = np.lexsort(
        (tie_break[released_mask], -jaccard[released_mask])
    )
    fake_order = np.lexsort(
        (tie_break[~released_mask], -jaccard[~released_mask])
    )
    selected = {
        candidate_edges[int(real_indices[index])]
        for index in real_order[:real_count]
    }
    selected.update(
        candidate_edges[int(fake_indices[index])]
        for index in fake_order[: target_count - real_count]
    )
    return selected


def edge_target_priority(edge: Edge, target_index: int) -> bytes:
    payload = (
        f"incidence:{edge[0]}:{edge[1]}:{target_index}"
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=8).digest()


def topology_features(
    targets: list[Edge],
    edges: list[Edge],
    node_count: int,
    incidence_cap: int,
) -> np.ndarray:
    targets_by_node: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        targets_by_node[target[0]].append(index)
        targets_by_node[target[1]].append(index)
    supports: list[set[int]] = [set() for _ in targets]
    for edge in edges:
        left, right = canonical(edge)
        eligible = sorted(
            set(targets_by_node[left]) | set(targets_by_node[right])
        )
        eligible = [
            index
            for index in eligible
            if {targets[index][0], targets[index][1]}
            != {left, right}
        ]
        eligible.sort(
            key=lambda index: edge_target_priority(
                (left, right),
                index,
            )
        )
        for index in eligible[:incidence_cap]:
            supports[index].add(left * node_count + right)
            supports[index].add(right * node_count + left)

    rows = []
    denominator = np.log1p(4 * incidence_cap + 1)
    for target, support in zip(targets, supports):
        u, v = target
        out_u: set[int] = set()
        in_u: set[int] = set()
        out_v: set[int] = set()
        in_v: set[int] = set()
        context_size = len(support)
        for code in support:
            a, b = divmod(code, node_count)
            if a == u:
                out_u.add(b)
            if b == u:
                in_u.add(a)
            if a == v:
                out_v.add(b)
            if b == v:
                in_v.add(a)
        common_out = len(out_u.intersection(in_v))
        common_in = len(in_u.intersection(out_v))
        common_norm = common_out / np.sqrt(
            max(len(out_u), 1) * max(len(in_v), 1)
        )
        rows.append(
            [
                np.log1p(context_size) / denominator,
                np.log1p(len(out_u)) / denominator,
                np.log1p(len(in_u)) / denominator,
                np.log1p(len(out_v)) / denominator,
                np.log1p(len(in_v)) / denominator,
                common_out / max(incidence_cap, 1),
                common_in / max(incidence_cap, 1),
                common_norm,
                float(context_size > 0),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def posterior_and_weights(
    released: np.ndarray,
    epsilon: float,
    positive_weight: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    q = 1.0 / (1.0 + math.exp(epsilon))
    prior = float((released.mean() - q) / (p - q))
    prior = float(np.clip(prior, 1e-6, 1.0 - 1e-6))
    posterior = np.empty_like(released, dtype=np.float32)
    positive = released == 1
    posterior[positive] = (
        p * prior / (p * prior + q * (1.0 - prior))
    )
    posterior[~positive] = (
        (1.0 - p) * prior
        / ((1.0 - p) * prior + (1.0 - q) * (1.0 - prior))
    )
    weights = (
        posterior * positive_weight + (1.0 - posterior)
    ).astype(np.float32)
    return posterior, weights


def train_model(
    train_features: np.ndarray,
    test_features: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
    model_kind: str,
) -> np.ndarray:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if model_kind == "f3":
        model: nn.Module = nn.Sequential(
            nn.Linear(train_features.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    elif model_kind == "linear":
        model = nn.Linear(train_features.shape[1], 1)
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_x = torch.from_numpy(train_features)
    test_x = torch.from_numpy(test_features)
    target_tensor = torch.from_numpy(target)
    weight_tensor = torch.from_numpy(sample_weights)
    for _ in range(2000):
        indices = rng.choice(
            len(train_features),
            size=128,
            replace=False,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x[indices]).view(-1)
        losses = (
            F.softplus(logits)
            - target_tensor[indices] * logits
        )
        loss = (losses * weight_tensor[indices]).mean()
        loss.backward()
        optimizer.step()
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(test_features), 512):
            outputs.extend(
                model(test_x[start : start + 512]).view(-1).tolist()
            )
    return np.asarray(outputs, dtype=np.float64)


def metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
    }


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    graph, raw_features = load_graph_data(args.data)
    true_edges = graph_edges(graph)
    release_candidates = feature_candidates(
        raw_features,
        args.release_candidates,
    )
    target_pairs = feature_candidates(
        raw_features,
        args.target_candidates,
    )
    workload = {
        name: [
            pair
            for pair in target_pairs
            if split_name(split_bucket(pair, args.split_salt)) == name
        ]
        for name in ("train", "valid", "test")
    }
    true_labels = {
        name: np.asarray(
            [pair in true_edges for pair in values],
            dtype=np.float32,
        )
        for name, values in workload.items()
    }
    projected = public_projected_features(raw_features)
    norms = np.sqrt(
        np.asarray(
            raw_features.multiply(raw_features).sum(axis=1)
        ).ravel()
    ).astype(np.float32)
    public = {
        name: pair_features_for(
            raw_features,
            projected,
            norms,
            values,
        )
        for name, values in workload.items()
    }

    runs = []
    for draw in range(args.release_draws):
        release_seed = args.base_seed + 101 * draw
        released = randomized_response_graph(
            release_candidates,
            true_edges,
            args.epsilon,
            release_seed,
        )
        denoised = jaccard_denoise(
            release_candidates,
            released,
            graph.shape[0],
            args.target_degree,
            args.epsilon,
            release_seed + 17,
        )
        topology = {
            name: topology_features(
                workload[name],
                sorted(denoised),
                graph.shape[0],
                args.incidence_cap,
            )[:, : args.topology_dimension]
            for name in ("train", "test")
        }
        train_bits = np.asarray(
            [pair in released for pair in workload["train"]],
            dtype=np.float32,
        )
        posterior, weights = posterior_and_weights(
            train_bits,
            args.epsilon,
        )
        joint = {
            name: np.concatenate(
                [public[name], topology[name]],
                axis=1,
            )
            for name in ("train", "test")
        }
        public_scores = train_model(
            public["train"],
            public["test"],
            posterior,
            weights,
            release_seed + 201,
            "f3",
        )
        joint_scores = train_model(
            joint["train"],
            joint["test"],
            posterior,
            weights,
            release_seed + 201,
            "f3",
        )
        train_permutation = np.random.default_rng(
            args.placebo_seed - 1000 + 101 * draw
        ).permutation(len(workload["train"]))
        test_permutation = np.random.default_rng(
            args.placebo_seed + 101 * draw
        ).permutation(len(workload["test"]))
        placebo = {
            "train": np.concatenate(
                [
                    public["train"],
                    topology["train"][train_permutation],
                ],
                axis=1,
            ),
            "test": np.concatenate(
                [public["test"], topology["test"][test_permutation]],
                axis=1,
            ),
        }
        placebo_scores = train_model(
            placebo["train"],
            placebo["test"],
            posterior,
            weights,
            release_seed + 201,
            "f3",
        )
        run_result = {
            "draw": draw,
            "release_edges": len(released),
            "denoised_edges": len(denoised),
            "public": metrics(public_scores, true_labels["test"]),
            "joint": metrics(joint_scores, true_labels["test"]),
            "placebo": metrics(placebo_scores, true_labels["test"]),
        }
        run_result["joint_minus_public"] = (
            run_result["joint"]["auc"]
            - run_result["public"]["auc"]
        )
        run_result["joint_minus_placebo"] = (
            run_result["joint"]["auc"]
            - run_result["placebo"]["auc"]
        )
        runs.append(run_result)
        print(json.dumps(run_result, ensure_ascii=False), flush=True)

    return {
        "dataset": args.dataset,
        "split_salt": args.split_salt,
        "epsilon": args.epsilon,
        "delta": 0.0,
        "release_draws": args.release_draws,
        "topology_dimension": args.topology_dimension,
        "label_counts": {
            name: {
                "total": int(len(values)),
                "positive": int(values.sum()),
            }
            for name, values in true_labels.items()
        },
        "runs": runs,
        "summary": {
            model: {
                "auc": summarize(
                    [run[model]["auc"] for run in runs]
                ),
                "average_precision": summarize(
                    [
                        run[model]["average_precision"]
                        for run in runs
                    ]
                ),
            }
            for model in ("public", "joint", "placebo")
        }
        | {
            "joint_minus_public": summarize(
                [run["joint_minus_public"] for run in runs]
            ),
            "joint_minus_placebo": summarize(
                [run["joint_minus_placebo"] for run in runs]
            ),
            "joint_above_placebo": int(
                sum(
                    run["joint_minus_placebo"] > 0
                    for run in runs
                )
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-salt", type=int, required=True)
    parser.add_argument("--release-draws", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=3.5)
    parser.add_argument("--target-candidates", type=int, default=20)
    parser.add_argument("--release-candidates", type=int, default=400)
    parser.add_argument("--target-degree", type=float, default=4.0)
    parser.add_argument("--incidence-cap", type=int, default=16)
    parser.add_argument(
        "--topology-dimension",
        type=int,
        choices=[8, 9],
        default=8,
    )
    parser.add_argument("--base-seed", type=int, default=13000)
    parser.add_argument("--placebo-seed", type=int, default=22000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
