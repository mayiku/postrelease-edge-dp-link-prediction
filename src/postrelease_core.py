"""Pure post-processing operations with no graph-file access."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Edge = tuple[int, int]


def canonical(edge: Edge) -> Edge:
    return (min(edge), max(edge))


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
    selection: str = "fixed",
    validation_features: np.ndarray | None = None,
    validation_target: np.ndarray | None = None,
    validation_weights: np.ndarray | None = None,
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
    validation_x = (
        torch.from_numpy(validation_features)
        if validation_features is not None
        else None
    )
    validation_target_tensor = (
        torch.from_numpy(validation_target)
        if validation_target is not None
        else None
    )
    validation_weight_tensor = (
        torch.from_numpy(validation_weights)
        if validation_weights is not None
        else None
    )
    best_loss = math.inf
    best_state = None
    for step in range(1, 2001):
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
        if (
            selection == "released-validation"
            and validation_x is not None
            and validation_target_tensor is not None
            and validation_weight_tensor is not None
            and (step % 100 == 0 or step == 2000)
        ):
            with torch.no_grad():
                validation_logits = model(validation_x).view(-1)
                validation_losses = (
                    F.softplus(validation_logits)
                    - validation_target_tensor * validation_logits
                )
                current_loss = float(
                    (
                        validation_losses
                        * validation_weight_tensor
                    ).mean()
                )
            if current_loss < best_loss:
                best_loss = current_loss
                best_state = deepcopy(model.state_dict())
    if selection == "released-validation" and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(test_features), 512):
            outputs.extend(
                model(test_x[start : start + 512]).view(-1).tolist()
            )
    return np.asarray(outputs, dtype=np.float64)


def rank01(values: np.ndarray) -> np.ndarray:
    ranks = np.argsort(
        np.argsort(values, kind="stable"),
        kind="stable",
    ).astype(np.float64)
    return ranks / max(len(ranks) - 1, 1)
