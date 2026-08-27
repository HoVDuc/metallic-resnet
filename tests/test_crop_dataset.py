import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models import PairCropDataset


class _ArrayDataset:
    def __init__(self):
        y, x = np.indices((24, 30))
        image = np.stack((x, y, x + y), axis=2).astype(np.uint8)
        erased = image.copy()
        erased[18:20, 23:25] = 0
        mask = np.zeros((24, 30), dtype=np.uint8)
        mask[18:20, 23:25] = 1
        self.sample = image, erased, mask

    def __getitem__(self, _index):
        return tuple(value.copy() for value in self.sample)


def test_pair_crop_dataset_returns_fixed_tensor_shapes():
    dataset = PairCropDataset(_ArrayDataset(), [0], crop_size=12, seed=7)

    view_a, view_b, mask = dataset[0]

    assert view_a.shape == view_b.shape == (3, 12, 12)
    assert mask.shape == (12, 12)
    assert view_a.dtype == view_b.dtype == mask.dtype == torch.float32


def test_validation_crop_is_deterministic_across_epochs():
    dataset = PairCropDataset(_ArrayDataset(), [0], crop_size=8, seed=11, train=False)
    first = dataset[0]
    dataset.set_epoch(3)
    second = dataset[0]

    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_training_crop_changes_with_epoch_reproducibly():
    dataset = PairCropDataset(
        _ArrayDataset(), [0], crop_size=8, seed=13,
        biased_probability=0.0, train=True,
    )
    epoch_zero = dataset[0]
    dataset.set_epoch(1)
    epoch_one = dataset[0]
    dataset.set_epoch(0)
    epoch_zero_again = dataset[0]

    assert torch.equal(epoch_zero[0], epoch_zero_again[0])
    assert not torch.equal(epoch_zero[0], epoch_one[0])


def test_crops_per_sample_expands_length_and_validates_arguments():
    dataset = PairCropDataset(_ArrayDataset(), [0, 1], crop_size=8, seed=1, crops_per_sample=3)
    assert len(dataset) == 6
    with pytest.raises(ValueError, match="crop_size"):
        PairCropDataset(_ArrayDataset(), [0], crop_size=0, seed=1)
