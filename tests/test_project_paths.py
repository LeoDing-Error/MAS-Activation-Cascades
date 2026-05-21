from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import project_paths


class PrependSysPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self._saved_path))

    def test_returns_false_and_leaves_path_untouched_for_missing_dir(self) -> None:
        missing = Path(tempfile.gettempdir()) / "definitely-not-a-real-dir-xyz"
        before = list(sys.path)

        result = project_paths._prepend_sys_path(missing)

        self.assertFalse(result)
        self.assertEqual(sys.path, before)

    def test_inserts_existing_dir_at_front(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = project_paths._prepend_sys_path(Path(tmp))

            self.assertTrue(result)
            self.assertEqual(sys.path[0], str(Path(tmp).resolve()))

    def test_repeated_calls_dedupe_to_a_single_front_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            resolved = str(Path(tmp).resolve())
            project_paths._prepend_sys_path(Path(tmp))
            sys.path.append("some-other-entry")

            project_paths._prepend_sys_path(Path(tmp))

            self.assertEqual(sys.path[0], resolved)
            self.assertEqual(sys.path.count(resolved), 1)


class EnsureOnPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self._saved_path))

    def test_ensure_local_camel_prepends_when_checkout_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(project_paths, "CAMEL_ROOT", Path(tmp)):
                self.assertTrue(project_paths.ensure_local_camel_on_path())
                self.assertEqual(sys.path[0], str(Path(tmp).resolve()))

    def test_ensure_local_ta2_returns_false_when_checkout_missing(self) -> None:
        missing = Path(tempfile.gettempdir()) / "no-ta2-checkout-here"
        with patch.object(project_paths, "TA2_ROOT", missing):
            self.assertFalse(project_paths.ensure_local_ta2_on_path())


class ResolveRepoPathTests(unittest.TestCase):
    def test_returns_none_for_unknown_name(self) -> None:
        self.assertIsNone(project_paths.resolve_repo_path("nonsense"))

    def test_returns_none_when_mapped_checkout_is_absent(self) -> None:
        missing = Path(tempfile.gettempdir()) / "absent-camel"
        with patch.object(project_paths, "CAMEL_ROOT", missing):
            self.assertIsNone(project_paths.resolve_repo_path("camel"))

    def test_name_lookup_is_case_insensitive_and_returns_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(project_paths, "TA2_ROOT", Path(tmp)):
                self.assertEqual(project_paths.resolve_repo_path("TA2"), Path(tmp))


if __name__ == "__main__":
    unittest.main()
