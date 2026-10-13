import unittest

from flashengine.backends.flashinfer import paged_metadata


class FlashInferMetadataTests(unittest.TestCase):
    def test_page_boundary_and_noncontiguous_tables(self):
        indptr, indices, last = paged_metadata([[7, 3], [2, 9, 5], [0]], [256, 257, 1], 256)
        self.assertEqual(indptr.tolist(), [0, 1, 3, 4])
        self.assertEqual(indices.tolist(), [7, 2, 9, 0])
        self.assertEqual(last.tolist(), [256, 1, 1])

    def test_short_table_is_rejected(self):
        with self.assertRaises(ValueError):
            paged_metadata([[2]], [257], 256)

    def test_empty_context_does_not_read_padding_pages(self):
        indptr, indices, last = paged_metadata([[-1], [4]], [0, 7], 64)
        self.assertEqual(indptr.tolist(), [0, 0, 1])
        self.assertEqual(indices.tolist(), [4])
        self.assertEqual(last.tolist(), [0, 7])
