import os
import os.path
from queue import Queue
import unittest

import gffutils

from peaks2utr import prepare_argparser
from peaks2utr.annotations import AnnotationsPipeline, NoNearbyFeatures, PotentialUTRZeroCoverage
from peaks2utr.collections import BroadPeaksList, ZeroCoverageIntervalsDict, SPATTruncationPointsDict
from peaks2utr.models import UTR, FeatureDB

TEST_DIR = os.path.dirname(__file__)


class TestCase2(unittest.TestCase):
    def setUp(self):
        db_path = os.path.join(TEST_DIR, "case2.db")
        gffutils.create_db(os.path.join(TEST_DIR, "case2.gtf"), db_path, force=True)
        self.db = FeatureDB(db_path)
        self.coverage_gaps = ZeroCoverageIntervalsDict()
        self.truncation_points = SPATTruncationPointsDict()
        forward_peaks_filename = os.path.join(TEST_DIR, "forward_peaks.broadPeak")
        self.forward_peaks = BroadPeaksList(broadpeak_fn=forward_peaks_filename, strand="forward")
        reverse_peaks_filename = os.path.join(TEST_DIR, "reverse_peaks.broadPeak")
        self.reverse_peaks = BroadPeaksList(broadpeak_fn=reverse_peaks_filename, strand="reverse")
        argparser = prepare_argparser()
        self.args = argparser.parse_args(["", ""])
        self.args.gtf_in = True

    def tearDown(self):
        os.remove(os.path.join(TEST_DIR, "case2.db"))

    def test_gene_within_exon(self):
        expected_annotations = {"forward_peak_32164": None}
        pipeline = AnnotationsPipeline(self.forward_peaks, self.args, queue=Queue())
        for peak in self.forward_peaks:
            if peak.name in expected_annotations:
                pipeline.annotate_utr_for_peak(self.db, peak, self.truncation_points, self.coverage_gaps)
                if expected_annotations[peak.name] is None:
                    self.assertIsNone(pipeline.queue.get())
                elif expected_annotations[peak.name] is NoNearbyFeatures:
                    self.assertIsInstance(pipeline.queue.get(), NoNearbyFeatures)
                elif expected_annotations[peak.name] is PotentialUTRZeroCoverage:
                    self.assertIsInstance(pipeline.queue.get(), PotentialUTRZeroCoverage)
                else:
                    result = None
                    annotations = AnnotationsDict()
                    while not pipeline.queue.empty():
                        result = pipeline.queue.get()
                        if type(result) == dict:
                            annotations.update(result)
                    for gene in expected_annotations[peak.name].keys():
                        self.assertIn(gene, annotations)
                        self.assertEqual(annotations.data[gene]['utr'].range, expected_annotations[peak.name][gene].range)

    def test_override_utr(self):
        self.args.max_distance = 5000
        self.args.override_utr = True
        expected_annotations = {
            "reverse_peak_32037": {"ENSMUSG00000033396": UTR(122048356, 122053879)},
            "forward_peak_32170": {"ENSMUSG00000027236": UTR(122052099, 122053546)}
        }
        pipeline = AnnotationsPipeline(self.forward_peaks, self.args, queue=Queue())
        for peak in self.forward_peaks + self.reverse_peaks:
            if peak.name in expected_annotations:
                pipeline.annotate_utr_for_peak(self.db, peak, self.truncation_points, self.coverage_gaps)
                result = pipeline.queue.get()
                if result:
                    for gene in expected_annotations[peak.name].keys():
                        self.assertIn(gene, result)
                        self.assertEqual(expected_annotations[peak.name][gene].range, result[gene]["utr"].range)
                else:
                    assert expected_annotations[peak.name] is None


if __name__ == '__main__':
    unittest.main()
