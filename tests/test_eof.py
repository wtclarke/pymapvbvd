"""Exercise EOF handling with synthetic complete and truncated VD/VB records."""

import struct
import tempfile
import unittest
import warnings

from mapvbvd.mapVBVD import loop_mdh_read
import numpy as np


def adc_record(version: str = "vd", samples: int = 4) -> bytes:
    header_size = 192 if version == "vd" else 128
    channel_header = 32 if version == "vd" else 0
    sample_offset = 48 if version == "vd" else 28
    record = bytearray(header_size + channel_header + samples * 8)
    struct.pack_into("<I", record, 0, len(record))
    struct.pack_into("<HH", record, sample_offset, samples, 1)
    return bytes(record)


def read_records(
    data: bytes, version: str = "vd", boundary: int | None = None, offset: int = 0
) -> tuple[np.ndarray, np.ndarray, bool]:
    with tempfile.TemporaryFile() as stream:
        stream.write(bytes(offset) + data)
        stream.seek(offset)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mdh, positions, eof = loop_mdh_read(
                stream, version, 1, 0, offset, len(data) if boundary is None else boundary, print_prog=False
            )
            return mdh, positions, bool(eof)


class ReaderEofTest(unittest.TestCase):
    def test_complete_adc_survives_eof_and_short_trailer(self) -> None:
        for version in ("vd", "vb"):
            adc = adc_record(version)
            for trailer in (b"", bytes(64)):
                mdh, positions, eof = read_records(adc * 2 + trailer, version)
                self.assertEqual(mdh.shape[1], 2)
                self.assertEqual(positions.tolist(), [0, len(adc)])
                self.assertTrue(eof)

    def test_truncated_adc_payload_is_not_accepted(self) -> None:
        for version in ("vd", "vb"):
            adc = adc_record(version)
            mdh, positions, eof = read_records(adc + adc[:-1], version)
            self.assertEqual(mdh.shape[1], 1)
            self.assertEqual(positions.tolist(), [0])
            self.assertTrue(eof)

    def test_truncated_next_header_does_not_remove_complete_adc(self) -> None:
        adc = adc_record()
        self.assertEqual(read_records(adc + adc[:100])[0].shape[1], 1)

    def test_measurement_boundary_excludes_next_measurement(self) -> None:
        adc = adc_record()
        mdh, positions, _ = read_records(adc * 3, boundary=len(adc) * 2)
        self.assertEqual(mdh.shape[1], 2)
        self.assertEqual(positions.tolist(), [0, len(adc)])

    def test_nonzero_measurement_offset(self) -> None:
        for version in ("vd", "vb"):
            adc = adc_record(version)
            mdh, positions, eof = read_records(adc * 3, version, boundary=2 * len(adc), offset=512)
            self.assertEqual(mdh.shape[1], 2)
            self.assertEqual(positions.tolist(), [512, 512 + len(adc)])
            self.assertTrue(eof)

    def test_measurement_boundary_excludes_crossing_payload(self) -> None:
        adc = adc_record()
        self.assertEqual(read_records(adc * 3, boundary=len(adc) * 2 - 1)[0].shape[1], 1)

    def test_acqend_preserves_preceding_adc(self) -> None:
        adc = adc_record()
        end = bytearray(192)
        struct.pack_into("<I", end, 0, 192)
        end[40] = 1
        mdh, _, eof = read_records(adc + end)
        self.assertEqual(mdh.shape[1], 1)
        self.assertFalse(eof)

    def test_no_complete_adc_returns_empty_arrays(self) -> None:
        for data in (b"", adc_record()[:-1]):
            mdh, positions, _ = read_records(data)
            self.assertEqual(mdh.shape[1], 0)
            self.assertEqual(positions.size, 0)


if __name__ == "__main__":
    unittest.main()
