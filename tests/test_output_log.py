"""
test_output_log.py

Unit tests for the anonymized inference log: confirm that logged records
match the frozen schema and never contain facial image data.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.retailvision.output_log import SCHEMA_FIELDS, log_detection

# A detection dict shaped like InferencePipeline.process_frame() output,
# including a real bbox/confidence to prove those get dropped before logging.
SAMPLE_DETECTION = {
    "bbox": (10, 20, 100, 100),
    "age_group": "18-40",
    "gender": "Male",
    "emotion": "Happy",
    "confidence": {"age": 0.91, "gender": 0.98, "emotion": 0.77},
}


class TestOutputLog(unittest.TestCase):
    """Verify log_detection() writes anonymized, schema-conforming records."""

    def setUp(self) -> None:
        """Point each test at a fresh temp log file."""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "inference_log.json"

    def tearDown(self) -> None:
        """Clean up the temp log file."""
        self.tmpdir.cleanup()

    def _read_records(self) -> list[dict]:
        """Parse every newline-delimited JSON record written to the log."""
        lines = self.log_path.read_text().splitlines()
        return [json.loads(line) for line in lines]

    def test_record_matches_frozen_schema(self) -> None:
        """A logged record has exactly the 8 frozen schema fields, nothing else."""
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        record = self._read_records()[0]
        self.assertEqual(set(record.keys()), set(SCHEMA_FIELDS))

    def test_record_contains_only_anonymized_labels(self) -> None:
        """Demographic/emotion labels round-trip correctly into the log."""
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        record = self._read_records()[0]
        self.assertEqual(record["age_group"], "18-40")
        self.assertEqual(record["gender"], "Male")
        self.assertEqual(record["emotion"], "Happy")

    def test_no_facial_image_data_in_log(self) -> None:
        """Neither bbox pixel coordinates nor classifier confidence reach the log."""
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        raw_text = self.log_path.read_text()
        record = self._read_records()[0]

        self.assertNotIn("bbox", record)
        self.assertNotIn("confidence", record)
        self.assertNotIn("bbox", raw_text)
        self.assertNotIn("confidence", raw_text)

    def test_unbuilt_fields_are_null_placeholders(self) -> None:
        """zone_id/count/dwell_seconds/engagement_score are null until their epics land."""
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        record = self._read_records()[0]
        for field in ("zone_id", "count", "dwell_seconds", "engagement_score"):
            self.assertIsNone(record[field])

    def test_append_mode_adds_one_record_per_call(self) -> None:
        """Each log_detection() call appends a new line, not overwriting prior records."""
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        log_detection(SAMPLE_DETECTION, log_path=self.log_path)
        self.assertEqual(len(self._read_records()), 2)


if __name__ == "__main__":
    unittest.main()
