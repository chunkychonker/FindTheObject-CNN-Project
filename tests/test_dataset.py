import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from findobject.data.dataset import Sample, SignScenes

N, S, H = 6, 16, 4


def _arrays(n: int = N) -> dict:
    rs = np.random.default_rng(0)
    return {
        "images": rs.integers(0, 256, (n, S, S, 3), dtype=np.uint8),
        "class_ids": rs.integers(0, 43, n).astype(np.int64),
        "boxes": np.tile(np.array([2, 3, 10, 12], np.int16), (n, 1)),
        "heatmaps": rs.random((n, H, H)).astype(np.float16),
    }


def _write(tmp_path, **arrays):
    path = tmp_path / "split.npz"
    np.savez(path, **arrays)
    return path


def test_len_and_field_types_shapes_ranges(tmp_path):
    ds = SignScenes(_write(tmp_path, **_arrays()))
    assert len(ds) == N
    s = ds[0]
    assert s.image.dtype == torch.float32 and s.image.shape == (3, S, S)
    assert 0 <= s.image.min() and s.image.max() <= 1
    assert s.heatmap.dtype == torch.float32 and s.heatmap.shape == (1, H, H)
    assert s.class_id.dtype == torch.int64 and s.class_id.shape == ()
    assert s.box.dtype == torch.int64 and s.box.tolist() == [2, 3, 10, 12]


def test_image_is_channels_first_and_round_trips_to_stored_pixels(tmp_path):
    arrays = _arrays()
    ds = SignScenes(_write(tmp_path, **arrays))
    for i in range(N):
        back = (ds[i].image * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()  # CHW -> HWC
        assert np.array_equal(back, arrays["images"][i])
        assert ds[i].class_id.item() == arrays["class_ids"][i]
        assert np.allclose(ds[i].heatmap[0].numpy(), arrays["heatmaps"][i].astype(np.float32))


def test_dataloader_batches_into_a_sample(tmp_path):
    ds = SignScenes(_write(tmp_path, **_arrays()))
    batch = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    assert isinstance(batch, Sample)
    assert batch.image.shape == (4, 3, S, S) and batch.heatmap.shape == (4, 1, H, H)
    assert batch.class_id.shape == (4,) and batch.box.shape == (4, 4)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SignScenes(tmp_path / "nope.npz")


def test_missing_array_raises(tmp_path):
    arrays = _arrays()
    del arrays["boxes"]
    with pytest.raises(KeyError):
        SignScenes(_write(tmp_path, **arrays))


def test_mismatched_lengths_raise(tmp_path):
    arrays = _arrays()
    arrays["class_ids"] = arrays["class_ids"][:-1]
    with pytest.raises(ValueError):
        SignScenes(_write(tmp_path, **arrays))


def test_out_of_range_class_id_raises(tmp_path):
    arrays = _arrays()
    arrays["class_ids"][0] = 43
    with pytest.raises(ValueError):
        SignScenes(_write(tmp_path, **arrays))


def test_index_out_of_range_raises(tmp_path):
    ds = SignScenes(_write(tmp_path, **_arrays()))
    with pytest.raises(IndexError):
        ds[N]
