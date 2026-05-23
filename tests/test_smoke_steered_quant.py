from __future__ import annotations

import importlib.util
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import smoke_steered_quant as smoke


class SmokeSteeredQuantDiagnosticsTests(unittest.TestCase):
    def test_package_status_reports_present_and_missing_packages(self) -> None:
        def fake_find_spec(name: str):
            if name == "gptqmodel":
                return object()
            return None

        with patch.object(importlib.util, "find_spec", side_effect=fake_find_spec):
            status = smoke.package_status(["gptqmodel", "optimum"])

        self.assertEqual(status, {"gptqmodel": True, "optimum": False})

    def test_cuda_summary_reports_unavailable_cuda(self) -> None:
        with patch.object(smoke.torch.cuda, "is_available", return_value=False):
            self.assertEqual(smoke.cuda_summary(), "cuda_available=False")

    def test_cuda_summary_reports_device_capability(self) -> None:
        props = SimpleNamespace(name="NVIDIA RTX PRO 6000 Blackwell")
        with patch.object(smoke.torch.cuda, "is_available", return_value=True), patch.object(
            smoke.torch.cuda, "current_device", return_value=0
        ), patch.object(smoke.torch.cuda, "get_device_properties", return_value=props), patch.object(
            smoke.torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            summary = smoke.cuda_summary()

        self.assertIn("cuda_available=True", summary)
        self.assertIn("device=0", summary)
        self.assertIn("capability=sm_120", summary)
        self.assertIn("NVIDIA RTX PRO 6000 Blackwell", summary)

    def test_print_runtime_context_includes_model_and_packages(self) -> None:
        stream = io.StringIO()
        expected_packages = ["gptqmodel", "optimum", "awq", "autoawq", "auto_awq"]
        with patch.object(smoke, "cuda_summary", return_value="cuda_available=False"), patch.object(
            smoke, "package_status", return_value={"gptqmodel": True, "optimum": False}
        ) as package_status, patch("sys.stdout", stream):
            smoke.print_runtime_context("hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4")

        package_status.assert_called_once_with(expected_packages)
        output = stream.getvalue()
        self.assertIn("model: hugging-quants/Meta-Llama-3.1-70B-Instruct-GPTQ-INT4", output)
        self.assertIn("cuda: cuda_available=False", output)
        self.assertIn("package gptqmodel: present", output)
        self.assertIn("package optimum: missing", output)


if __name__ == "__main__":
    unittest.main()
