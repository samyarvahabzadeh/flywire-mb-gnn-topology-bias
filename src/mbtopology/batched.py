"""Independent GCN/GAT models scheduled on one CUDA device."""

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

import torch
from torch.nn import functional as F

from . import models
from .experiment_io import (
    SessionYield,
    load_training_checkpoint,
    save_training_checkpoint,
)
from .runtime import require

PILOT_SEED = 1337


def precompute_features(data):
    return torch.stack(
        [models.node_features(data["roles"], data["kc"], pattern) for pattern in data["patterns"]]
    )


def probe_values(model, data, features):
    if model.architecture == "GAT":
        output = torch.stack([model(cue, data["graph"]) for cue in features])
    else:
        output = model(features, data["graph"])
    mbons = output[:, data["mbon"]]
    avoidance = mbons[:, data["avoidance"]].mean(dim=1)
    approach = mbons[:, data["approach"]].mean(dim=1)
    return ((avoidance - approach + 1) / 2).clamp(0, 1)


@dataclass
class BatchJob:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    data: dict
    checkpoint_dir: Path
    signature: str
    stream: torch.cuda.Stream
    features: torch.Tensor
    history: list = field(default_factory=list)
    streak: int = 0
    reached_at: int | None = None

    def restore(self):
        saved = load_training_checkpoint(self.checkpoint_dir, self.signature)
        if saved is not None:
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.history = saved["history"]
            self.streak = saved["streak"]
            self.reached_at = saved["reached_at"]

    def checkpoint(self):
        save_training_checkpoint(
            self.checkpoint_dir,
            self.signature,
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
                "cursor": len(self.history),
                "streak": self.streak,
                "reached_at": self.reached_at,
            },
        )

    def enqueue(self, task):
        odor = int(self.data["schedule"][len(self.history)])
        positive = odor in task["task"]["cs_plus"]
        with torch.cuda.stream(self.stream):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(self.features[odor], self.data["graph"])
            value = models.valence_readout(
                output, self.data["mbon"], self.data["avoidance"], self.data["approach"]
            )
            loss = F.binary_cross_entropy(value, value.new_tensor(float(positive)))
            loss.backward()
            finite_grad = torch.stack(
                [
                    torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                    if p.grad is not None
                ]
            ).all()
            self.optimizer.step()
            self.model.eval()
            with torch.no_grad():
                values = probe_values(self.model, self.data, self.features).double()
                plus = values[task["task"]["cs_plus"]].mean()
                minus = values[task["task"]["cs_minus"]].mean()
                score = (plus + 1 - minus) / 2
                valid = finite_grad & torch.isfinite(loss) & torch.isfinite(score)
                observation = torch.stack((loss.detach().double(), score, valid.double()))
        return odor, int(positive), observation


def train_batch(jobs, task, trials, deadline, *, stop_at_criterion, checkpoint_seconds=900):
    """Enqueue each active job before synchronizing and committing the full round."""
    require(0 < trials <= task["task"]["trials_max"], "invalid batch trial budget")
    require(0 < checkpoint_seconds <= 900, "invalid batch checkpoint interval")
    default = torch.cuda.current_stream()
    for job in jobs:
        job.restore()
        require(len(job.history) <= trials, "batch checkpoint exceeds requested trial budget")
        job.stream.wait_stream(default)
    last_save = monotonic()

    def save():
        for job in jobs:
            job.checkpoint()

    def active(job):
        return len(job.history) < trials and (job.reached_at is None or not stop_at_criterion)

    while pending := [job for job in jobs if active(job)]:
        if monotonic() >= deadline:
            save()
            raise SessionYield("batch checkpointed at the session deadline")
        queued = [job.enqueue(task) for job in pending]
        for job in pending:
            default.wait_stream(job.stream)
        observations = torch.stack([item[2] for item in queued]).cpu().tolist()
        require(all(item[2] for item in observations), "nonfinite batch output or gradient")
        for job, (odor, us, _), (loss, score, _) in zip(pending, queued, observations, strict=True):
            trial = len(job.history) + 1
            job.streak = job.streak + 1 if score >= task["readout"]["criterion"] else 0
            if job.reached_at is None and job.streak >= task["readout"]["consecutive_trials"]:
                job.reached_at = trial
            job.history.append(
                {
                    "trial": trial,
                    "odor_id": odor,
                    "us": us,
                    "training_loss": loss,
                    "discrimination": score,
                    "criterion_streak": job.streak,
                }
            )
        if monotonic() - last_save >= checkpoint_seconds:
            save()
            last_save = monotonic()
    torch.cuda.synchronize()
    save()
