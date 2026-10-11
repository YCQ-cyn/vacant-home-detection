from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import ROOT, SEED, bool_col, key, write_json


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = (20260926, 20260927, 20260928)
WINDOWS = (14, 21, 28, 42)
torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4 if width >= 4 else 1, width),
            nn.GELU(),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4 if width >= 4 else 1, width),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class DailyDualTCN(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.behavior_channels = (1, 2, 3, 4, 5, 6, 7)
        self.magnitude_channels = (0, 4, 5, 6, 7)
        self.behavior_in = nn.Conv1d(len(self.behavior_channels), hidden, 1)
        self.magnitude_in = nn.Conv1d(len(self.magnitude_channels), hidden, 1)
        self.behavior_blocks = nn.Sequential(*(ResidualBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.magnitude_blocks = nn.Sequential(*(ResidualBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.behavior_project = nn.Conv1d(hidden, 32, 1)
        self.magnitude_project = nn.Conv1d(hidden, 32, 1)
        self.decoder = nn.Conv1d(64, 1, 1)
        self.magnitude_head = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 3))

    def forward(self, x):
        behavior_sequence = self.behavior_project(self.behavior_blocks(self.behavior_in(x[:, self.behavior_channels])))
        magnitude_sequence = self.magnitude_project(self.magnitude_blocks(self.magnitude_in(x[:, self.magnitude_channels])))
        sequence = torch.cat([behavior_sequence, magnitude_sequence], dim=1)
        embedding = sequence.mean(-1)
        behavior_embedding = behavior_sequence.mean(-1)
        magnitude_embedding = magnitude_sequence.mean(-1)
        reconstruction = self.decoder(sequence).squeeze(1)
        magnitude = self.magnitude_head(magnitude_embedding)
        return embedding, behavior_embedding, magnitude_embedding, sequence, reconstruction, magnitude


def mask_blocks(x: torch.Tensor, generator: torch.Generator):
    view = x.clone()
    mask = torch.zeros((len(x), x.shape[-1]), dtype=torch.bool, device=x.device)
    starts = torch.randint(0, x.shape[-1] - 16, (len(x),), generator=generator)
    widths = torch.randint(8, 21, (len(x),), generator=generator)
    for row, (start, width) in enumerate(zip(starts.tolist(), widths.tolist())):
        mask[row, start:min(x.shape[-1], start + width)] = True
    view = view.masked_fill(mask[:, None, :], 0)
    view[:, 4] = torch.maximum(view[:, 4], mask.float())
    return view, mask


def info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = .2):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def hierarchical_contextual_loss(sequence_a: torch.Tensor, sequence_b: torch.Tensor):
    """TS2Vec-style hierarchy without clock-time destroying shifts or warps."""
    a = sequence_a.transpose(1, 2)
    b = sequence_b.transpose(1, 2)
    losses = []
    while a.shape[1] >= 4:
        # Instance discrimination uses the same user's aligned context as the positive.
        losses.append(info_nce(a.mean(1), b.mean(1)))
        # Timestamp discrimination keeps exact clock alignment and samples bounded positions.
        positions = torch.linspace(0, a.shape[1] - 1, min(12, a.shape[1]), device=a.device).long()
        temporal_a = a[:, positions].reshape(-1, a.shape[-1])
        temporal_b = b[:, positions].reshape(-1, b.shape[-1])
        losses.append(.5 * info_nce(temporal_a, temporal_b))
        a = F.max_pool1d(a.transpose(1, 2), kernel_size=2).transpose(1, 2)
        b = F.max_pool1d(b.transpose(1, 2), kernel_size=2).transpose(1, 2)
    return torch.stack(losses).mean()


def load_inputs():
    x = np.load(ROOT / "inputs" / "daily_inputs.npz")["x"].astype(np.float32)
    index = pd.read_csv(ROOT / "inputs" / "daily_input_index.csv", dtype={"meter_id": str})
    index["meter_id"] = index.meter_id.map(key)
    index["date"] = pd.to_datetime(index.date)
    scope = pd.read_csv(ROOT / "fit_scope.csv", dtype={"meter_id": str})
    scope["meter_id"] = scope.meter_id.map(key)
    for column in ["research_fit", "final_fit", "ssl_fit", "future_holdout", "strict_all_zero"]:
        scope[column] = bool_col(scope[column])
    return x, index, scope


def balanced_indices(index, fit_ids, zero_ids, zero_ratio, seed, maximum=None):
    fit = index.meter_id.isin(fit_ids).to_numpy()
    zero = index.meter_id.isin(zero_ids).to_numpy() & fit
    active = np.where(fit & ~zero)[0]
    silent = np.where(zero)[0]
    rng = np.random.default_rng(seed)
    n_silent = min(len(silent), int(len(active) * zero_ratio / max(1e-6, 1 - zero_ratio)))
    selected = np.r_[active, rng.choice(silent, n_silent, replace=False)]
    if maximum and len(selected) > maximum:
        selected = rng.choice(selected, maximum, replace=False)
    rng.shuffle(selected)
    return selected


def train_daily(scope_name, fit_ids, hidden, zero_ratio, contrastive, seed, epochs, maximum=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x, index, scope = load_inputs()
    holdout = set(scope.loc[scope.future_holdout, "meter_id"])
    if set(fit_ids) & holdout:
        raise RuntimeError("Future holdout leaked into Daily SSL")
    zero_ids = set(scope.loc[scope.strict_all_zero, "meter_id"])
    model = DailyDualTCN(hidden).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2e-3)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        selected = balanced_indices(index, fit_ids, zero_ids, zero_ratio, seed + epoch, maximum)
        model.train()
        losses = []
        components = []
        for start in range(0, len(selected), 192):
            xb = torch.from_numpy(x[selected[start:start + 192]]).to(DEVICE)
            view_a, mask = mask_blocks(xb, generator)
            view_b, _ = mask_blocks(xb, generator)
            z, behavior, magnitude, sequence, reconstruction, magnitude_prediction = model(view_a)
            _, _, _, sequence_b, _, _ = model(view_b)
            signed_log = xb[:, 0]
            reconstruction_loss = ((reconstruction - signed_log) ** 2)[mask].mean()
            magnitude_target = torch.stack([
                signed_log.median(1).values,
                torch.quantile(signed_log, .9, dim=1),
                signed_log.abs().amax(1),
            ], dim=1)
            magnitude_loss = F.smooth_l1_loss(magnitude_prediction, magnitude_target)
            # Recompute every amplitude-dependent channel seen by the behavior
            # branch. Changing channel 0 alone would make this objective vacuous,
            # because the behavior branch does not consume channel 0 directly.
            _, behavior_original, _, _, _, _ = model(xb)
            scaled = xb.clone()
            scale = torch.exp(torch.empty((len(xb), 1, 1), device=DEVICE).uniform_(-.8, .8))
            raw = torch.expm1(scaled[:, :1].abs()) * scale
            scaled[:, :1] = torch.sign(scaled[:, :1]) * torch.log1p(raw)
            scaled[:, 2, 0] = 0
            scaled[:, 2, 1:] = scaled[:, 0, 1:] - scaled[:, 0, :-1]
            _, behavior_scaled, _, _, _, _ = model(scaled)
            invariance_loss = 1 - F.cosine_similarity(behavior_original, behavior_scaled, dim=1).mean()
            contrastive_loss = hierarchical_contextual_loss(sequence, sequence_b) if contrastive else torch.tensor(0.0, device=DEVICE)
            variance_penalty = F.relu(.2 - z.std(0)).mean()
            loss = reconstruction_loss + .25 * magnitude_loss + .15 * invariance_loss + .10 * variance_penalty
            if contrastive:
                loss = loss + .08 * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            components.append([
                float(reconstruction_loss.detach().cpu()), float(magnitude_loss.detach().cpu()),
                float(invariance_loss.detach().cpu()), float(contrastive_loss.detach().cpu()),
            ])
        component_mean = np.mean(components, axis=0)
        history.append({
            "epoch": epoch + 1, "loss": float(np.mean(losses)), "days": int(len(selected)),
            "reconstruction": float(component_mean[0]), "magnitude": float(component_mean[1]),
            "behavior_invariance": float(component_mean[2]), "hierarchical_contextual": float(component_mean[3]),
        })
        print(json.dumps({"daily": scope_name, "seed": seed, **history[-1]}), flush=True)
    output = ROOT / "models" / "representations" / scope_name / f"daily_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "hidden": hidden, "zero_ratio": zero_ratio,
        "contrastive": contrastive, "seed": seed, "fit_meter_ids": sorted(fit_ids),
        "holdout_manifest_sha256": json.loads((ROOT / "future_human_holdout_freeze.json").read_text(encoding="utf-8"))["manifest_sha256"],
    }, output / "daily_encoder.pt")
    model.eval()
    inference_positions = np.arange(len(index))
    if scope_name.startswith("research"):
        inference_positions = np.where(~index.meter_id.isin(holdout).to_numpy())[0]
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(inference_positions), 512):
            positions = inference_positions[start:start + 512]
            embeddings.append(model(torch.from_numpy(x[positions]).to(DEVICE))[0].cpu().numpy())
    embeddings = np.concatenate(embeddings).astype(np.float32)
    np.save(output / "daily_embeddings.npy", embeddings)
    index.iloc[inference_positions][["meter_id", "date"]].reset_index(drop=True).to_csv(
        output / "daily_embedding_index.csv", index=False
    )
    write_json(output / "training_report.json", {
        "scope": scope_name, "fit_meter_count": len(fit_ids), "fit_day_cap": maximum,
        "hidden": hidden, "zero_ratio": zero_ratio, "contrastive": contrastive,
        "seed": seed, "device": str(DEVICE), "history": history,
        "future_holdout_overlap": len(set(fit_ids) & holdout),
    })
    return output, history[-1]


def build_user_sequences(embedding, index):
    meters = sorted(index.meter_id.unique())
    sequences = np.zeros((len(meters), 42, embedding.shape[1]), np.float32)
    masks = np.zeros((len(meters), 42), np.float32)
    for row, meter_id in enumerate(meters):
        positions = index.index[index.meter_id.eq(meter_id)].to_numpy()
        positions = positions[np.argsort(index.loc[positions, "date"].to_numpy())][-42:]
        sequences[row, -len(positions):] = embedding[positions]
        masks[row, -len(positions):] = 1
    return meters, sequences, masks


class LongEncoder(nn.Module):
    def __init__(self, kind="tcn"):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.encoder = nn.GRU(64, 64, batch_first=True)
        else:
            self.input = nn.Conv1d(64, 64, 1)
            self.encoder = nn.Sequential(*(ResidualBlock(64, d) for d in (1, 2, 4, 8)))
        self.gate = nn.Linear(64, 1)
        self.reconstruction = nn.Linear(64, 64)
        self.order = nn.Linear(64, 1)

    def forward(self, sequence, mask):
        if self.kind == "gru":
            hidden, _ = self.encoder(sequence)
        else:
            hidden = self.encoder(self.input(sequence.transpose(1, 2))).transpose(1, 2)
        attention = self.gate(hidden).squeeze(-1).masked_fill(mask <= 0, -1e4).softmax(1)
        pooled = (hidden * attention[:, :, None]).sum(1)
        return pooled, hidden, self.reconstruction(hidden), self.order(pooled).squeeze(1)


def train_long(scope_name, daily_output: Path, fit_ids, seed, kind="tcn", epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    embedding = np.load(daily_output / "daily_embeddings.npy")
    index = pd.read_csv(daily_output / "daily_embedding_index.csv", dtype={"meter_id": str})
    index["date"] = pd.to_datetime(index.date)
    meters, sequence, mask = build_user_sequences(embedding, index)
    scope = pd.read_csv(ROOT / "fit_scope.csv", dtype={"meter_id": str})
    scope["future_holdout"] = bool_col(scope.future_holdout)
    holdout = set(scope.loc[scope.future_holdout, "meter_id"])
    if set(fit_ids) & holdout:
        raise RuntimeError("Future holdout leaked into Long-Horizon SSL")
    fit_rows = np.array([i for i, meter in enumerate(meters) if meter in fit_ids])
    model = LongEncoder(kind).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-3)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        rng.shuffle(fit_rows)
        losses = []
        for start in range(0, len(fit_rows), 96):
            rows = fit_rows[start:start + 96]
            original = torch.from_numpy(sequence[rows]).to(DEVICE)
            valid = torch.from_numpy(mask[rows]).to(DEVICE)
            day_mask = (torch.rand((len(rows), 42), device=DEVICE) < .18) & (valid > 0)
            masked = original.masked_fill(day_mask[:, :, None], 0)
            pooled, hidden, reconstruction, order = model(masked, valid)
            reconstruction_loss = ((reconstruction - original) ** 2)[day_mask].mean()
            context_loss = torch.tensor(0.0, device=DEVICE)
            for width in (14, 21, 28):
                short_mask = valid.clone()
                short_mask[:, :-width] = 0
                short_embedding = model(masked * short_mask[:, :, None], short_mask)[0]
                context_loss = context_loss + (1 - F.cosine_similarity(pooled, short_embedding, dim=1).mean()) / 3
            reverse_embedding, _, _, reverse_order = model(torch.flip(masked, (1,)), torch.flip(valid, (1,)))
            order_loss = (
                F.binary_cross_entropy_with_logits(order, torch.ones_like(order)) +
                F.binary_cross_entropy_with_logits(reverse_order, torch.zeros_like(reverse_order))
            ) / 2
            separation = F.relu(.18 - pooled.std(0)).mean()
            loss = reconstruction_loss + .18 * context_loss + .08 * order_loss + .08 * separation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses))})
        if epoch in {0, epochs - 1} or (epoch + 1) % 5 == 0:
            print(json.dumps({"long": scope_name, "kind": kind, "seed": seed, **history[-1]}), flush=True)
    output = ROOT / "models" / "representations" / scope_name / f"long_{kind}_seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    outputs = []
    windows = {width: [] for width in WINDOWS}
    with torch.no_grad():
        for start in range(0, len(meters), 256):
            xb = torch.from_numpy(sequence[start:start + 256]).to(DEVICE)
            mb = torch.from_numpy(mask[start:start + 256]).to(DEVICE)
            outputs.append(model(xb, mb)[0].cpu().numpy())
            for width in WINDOWS:
                short_mask = mb.clone()
                short_mask[:, :-width] = 0
                windows[width].append(model(xb * short_mask[:, :, None], short_mask)[0].cpu().numpy())
    np.save(output / "user_embeddings.npy", np.concatenate(outputs).astype(np.float32))
    for width, pieces in windows.items():
        np.save(output / f"user_embeddings_window_{width}.npy", np.concatenate(pieces).astype(np.float32))
    pd.DataFrame({"meter_id": meters}).to_csv(output / "user_embedding_index.csv", index=False)
    torch.save({
        "state_dict": model.state_dict(), "kind": kind, "seed": seed,
        "fit_meter_ids": sorted(fit_ids), "daily_encoder_path": str(daily_output.relative_to(ROOT)),
    }, output / "user_encoder.pt")
    write_json(output / "training_report.json", {
        "scope": scope_name, "kind": kind, "seed": seed, "fit_meter_count": len(fit_rows),
        "future_holdout_overlap": len(set(fit_ids) & holdout), "history": history,
    })
    return output


def run_research(screen_only=False):
    _, _, scope = load_inputs()
    fit_ids = set(scope.loc[scope.research_fit, "meter_id"])
    if not (ROOT / "representation_screen_selection.json").exists():
        rows = []
        for hidden in (32, 64):
            for zero_ratio in (.10, .20):
                for contrastive in (False, True):
                    name = f"screen_h{hidden}_z{int(zero_ratio * 100)}_c{int(contrastive)}"
                    _, metrics = train_daily(name, fit_ids, hidden, zero_ratio, contrastive, SEEDS[0], 1, 20000)
                    rows.append({
                        "configuration": name, "hidden": hidden, "zero_ratio": zero_ratio,
                        "contrastive": contrastive, **metrics,
                    })
        screen = pd.DataFrame(rows)
        screen.to_csv(ROOT / "daily_ssl_screening.csv", index=False)
        # Hidden width and strict-zero ratio are selected on reconstruction+magnitude proxy only.
        screen["noncontrastive_proxy"] = screen.reconstruction + .25 * screen.magnitude
        grouped = screen.groupby(["hidden", "zero_ratio"], as_index=False).noncontrastive_proxy.mean()
        chosen = grouped.sort_values("noncontrastive_proxy").iloc[0]
        write_json(ROOT / "representation_screen_selection.json", {
            "hidden": int(chosen.hidden), "zero_ratio": float(chosen.zero_ratio),
            "contrastive_is_not_selected_on_ssl_loss": True,
            "contrastive_decision": "compare downstream plain versus contextual TCN using hard-gate scorecard",
        })
    selection = json.loads((ROOT / "representation_screen_selection.json").read_text(encoding="utf-8"))
    if screen_only:
        return
    for contrastive in (False, True):
        variant = "research_contrastive" if contrastive else "research_plain"
        for seed in SEEDS:
            daily_output, _ = train_daily(
                variant, fit_ids, int(selection["hidden"]), float(selection["zero_ratio"]), contrastive,
                seed, 6, None,
            )
            train_long(variant, daily_output, fit_ids, seed, "tcn", 25)
    # GRU is a single structural sensitivity using the plain daily representation.
    for seed in SEEDS:
        daily_output = ROOT / "models" / "representations" / "research_plain" / f"daily_seed_{seed}"
        train_long("research_gru", daily_output, fit_ids, seed, "gru", 25)


def sanitize_research_holdout_embeddings():
    """Discard any pre-freeze research inference rows from interrupted/older runs."""
    _, _, scope = load_inputs()
    holdout = set(scope.loc[scope.future_holdout, "meter_id"])
    removed = []
    representation_root = ROOT / "models" / "representations"
    variants = ["research_plain", "research_contrastive"] + [path.name for path in representation_root.glob("screen_*") if path.is_dir()]
    for variant in variants:
        variant_root = representation_root / variant
        for daily_output in variant_root.glob("daily_seed_*"):
            index_path = daily_output / "daily_embedding_index.csv"
            embedding_path = daily_output / "daily_embeddings.npy"
            if index_path.exists() and embedding_path.exists():
                index = pd.read_csv(index_path, dtype={"meter_id": str})
                embedding = np.load(embedding_path)
                keep = ~index.meter_id.isin(holdout).to_numpy()
                if (~keep).any():
                    removed.extend(index.loc[~keep, "meter_id"].tolist())
                    np.save(embedding_path, embedding[keep].astype(np.float32))
                    index.loc[keep].reset_index(drop=True).to_csv(index_path, index=False)
        for long_output in variant_root.glob("long_*_seed_*"):
            _sanitize_user_output(long_output, holdout, removed)
    for seed in SEEDS:
        _sanitize_user_output(ROOT / "models" / "representations" / "research_gru" / f"long_gru_seed_{seed}", holdout, removed)
    write_json(ROOT / "research_holdout_inference_sanitization.json", {
        "holdout_count": len(holdout),
        "removed_rows": len(removed),
        "unique_removed_meter_ids": len(set(removed)),
        "postcondition": "research representation files contain no future holdout meters",
    })


def _sanitize_user_output(output: Path, holdout, removed):
    index_path = output / "user_embedding_index.csv"
    if not index_path.exists():
        return
    index = pd.read_csv(index_path, dtype={"meter_id": str})
    keep = ~index.meter_id.isin(holdout).to_numpy()
    if not (~keep).any():
        return
    removed.extend(index.loc[~keep, "meter_id"].tolist())
    for path in output.glob("user_embeddings*.npy"):
        values = np.load(path)
        np.save(path, values[keep].astype(np.float32))
    index.loc[keep].reset_index(drop=True).to_csv(index_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--sanitize-research-holdout", action="store_true")
    args = parser.parse_args()
    if args.sanitize_research_holdout:
        sanitize_research_holdout_embeddings()
    else:
        run_research(args.screen_only)
