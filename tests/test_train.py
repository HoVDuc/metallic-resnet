import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn

from losses import MultiLayerDiffLoss, PosWeightScheduler
from train import _run_epoch, _run_full_resolution_epoch, _make_grad_scaler, build_parser


class _RecordingDifferenceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.input_shapes = []

    def forward(self, view_a, view_b):
        self.input_shapes.append(tuple(view_a.shape))
        difference = (view_a - view_b).abs().mean(dim=1)
        prediction = torch.sigmoid(self.scale * difference)
        return {"learned": {"f0": prediction}}


def _sample(height, width):
    original = np.zeros((height, width, 3), dtype=np.uint8)
    erased = original.copy()
    erased[height // 2, width // 2] = 255
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2, width // 2] = 1
    return original, erased, mask


def test_full_resolution_fallback_processes_different_shapes_sequentially():
    samples = [_sample(5, 7), _sample(8, 4)]
    originals, erased, masks = zip(*samples)
    loader = [(list(originals), list(erased), list(masks))]
    model = _RecordingDifferenceModel()
    criterion = MultiLayerDiffLoss({"f0": 1.0}, {"learned": 1.0})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = PosWeightScheduler("linear", alpha_max=2.0, hold_pairs=0, ramp_pairs=10)

    loss, components = _run_full_resolution_epoch(
        loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
        scaler=_make_grad_scaler(torch.device("cpu"), None),
        amp_dtype=None,
        trainable_params=list(model.parameters()),
    )

    assert model.input_shapes == [(1, 3, 5, 7), (1, 3, 8, 4)]
    assert loss > 0
    assert "learned/f0" in components
    assert scheduler.total_pairs == 2


def test_run_epoch_uses_one_forward_for_a_tensor_batch():
    first = _sample(8, 8)
    second = _sample(8, 8)
    from models import prepare_pair_batch

    view_a, view_b, masks = prepare_pair_batch(
        [first[0], second[0]], [first[1], second[1]], [first[2], second[2]]
    )
    model = _RecordingDifferenceModel()
    criterion = MultiLayerDiffLoss({"f0": 1.0}, {"learned": 1.0})
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = PosWeightScheduler("linear", alpha_max=2.0, hold_pairs=0, ramp_pairs=10)

    loss, _ = _run_epoch(
        [(view_a, view_b, masks)],
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        max_grad_norm=1.0,
    )

    assert model.input_shapes == [(2, 3, 8, 8)]
    assert loss > 0
    assert scheduler.total_pairs == 2


def test_train_cli_defaults_to_true_cropped_batches():
    args = build_parser().parse_args(["--root", "pairs"])

    assert args.crop_size == 512
    assert args.batch == 8
    assert args.workers == 4
    assert args.amp == "auto"
