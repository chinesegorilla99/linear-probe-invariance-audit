"""Tests for the artifact manifest's released/unreleased check semantics.

The manifest records artifacts the repository cannot all ship: encoder
checkpoints are 17 GB and stay out. --check must therefore fail on a released
file that is missing or altered and must not fail on an absent unreleased one,
because that distinction is what lets a reviewer run the check at all.

Run:  python -m unittest tests.test_manifest -v
"""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.eval import manifest


class TestManifestReleaseScope(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        subprocess.run(["git", "init", "-q"], cwd=self.tmp, check=True)
        (self.tmp / "configs").mkdir()
        (self.tmp / "configs" / "kept.yaml").write_text("a: 1\n")
        (self.tmp / "configs" / "ignored.yaml").write_text("b: 2\n")
        (self.tmp / ".gitignore").write_text("configs/ignored.yaml\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _entries(self):
        return json.loads((self.tmp / manifest.MANIFEST).read_text())["files"]

    def test_gitignore_decides_released(self):
        manifest.write(self.tmp)
        e = self._entries()
        self.assertTrue(e["configs/kept.yaml"]["released"])
        self.assertFalse(e["configs/ignored.yaml"]["released"])

    def test_absent_unreleased_file_passes(self):
        manifest.write(self.tmp)
        (self.tmp / "configs" / "ignored.yaml").unlink()
        self.assertEqual(manifest.check(self.tmp), 0)

    def test_missing_released_file_fails(self):
        manifest.write(self.tmp)
        (self.tmp / "configs" / "kept.yaml").unlink()
        self.assertEqual(manifest.check(self.tmp), 1)

    def test_altered_unreleased_file_still_fails(self):
        """Absent is excused; altered never is, released or not."""
        manifest.write(self.tmp)
        (self.tmp / "configs" / "ignored.yaml").write_text("b: 99\n")
        self.assertEqual(manifest.check(self.tmp), 1)

    def test_legacy_manifest_without_released_field(self):
        """A manifest written before the field existed treats every entry as released."""
        manifest.write(self.tmp)
        path = self.tmp / manifest.MANIFEST
        doc = json.loads(path.read_text())
        for entry in doc["files"].values():
            entry.pop("released")
        path.write_text(json.dumps(doc))
        (self.tmp / "configs" / "ignored.yaml").unlink()
        self.assertEqual(manifest.check(self.tmp), 1)


if __name__ == "__main__":
    unittest.main()
