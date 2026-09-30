import numpy as np

from mapvbvd.twix_map_obj import _store_unique_block


def test_store_unique_block_contiguous_and_scattered_indices():
    out = np.zeros((2, 2, 5), dtype=np.complex64)
    count = np.zeros((1, 1, 5), dtype=np.float32)
    block = np.arange(12, dtype=np.float32).reshape(2, 2, 3).astype(np.complex64)

    _store_unique_block(out, block, np.array([1, 2, 3]), count)
    np.testing.assert_array_equal(out[:, :, 1:4], block)
    np.testing.assert_array_equal(count[:, :, 1:4], 1)

    _store_unique_block(out, block, np.array([1, 2, 3]), count)
    np.testing.assert_array_equal(out[:, :, 1:4], 2 * block)

    _store_unique_block(out, block[:, :, :2], np.array([0, 4]), count)
    np.testing.assert_array_equal(out[:, :, [0, 4]], block[:, :, :2])
