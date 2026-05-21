from __future__ import annotations

import sys
import types
import unittest

# Stub heavy/external deps so steering_backend imports on CPU without GPU,
# camel, transformers, openai, or pydantic actually installed. We only exercise
# the pure _build_model_kwargs helper, so object placeholders are sufficient.
_STUBS = {
    "openai": {"AsyncStream": object, "Stream": object},
    "pydantic": {"BaseModel": object},
    "camel.messages": {"OpenAIMessage": object},
    "camel.models": {"BaseModelBackend": object},
    "camel.types": {
        "ChatCompletion": object,
        "ChatCompletionChunk": object,
        "ChatCompletionMessage": object,
        "Choice": object,
        "CompletionUsage": object,
    },
    "camel.utils": {"BaseTokenCounter": object},
    "transformers": {
        "AutoModelForCausalLM": object,
        "AutoTokenizer": object,
        "AutoConfig": object,
    },
}
for _name, _attrs in _STUBS.items():
    _module = types.ModuleType(_name)
    for _attr, _value in _attrs.items():
        setattr(_module, _attr, _value)
    sys.modules[_name] = _module
sys.modules.setdefault("camel", types.ModuleType("camel"))

# src/backends/__init__.py eagerly imports camel_integration (the full camel
# agent stack). Stub that submodule so importing the package only loads the
# steering_backend module we are testing.
_fake_ci = types.ModuleType("src.backends.camel_integration")
for _ci_name in (
    "AgentSpec",
    "create_chat_agent",
    "create_clean_chat_agent",
    "create_openai_compatible_agent",
    "create_role_playing_session",
):
    setattr(_fake_ci, _ci_name, object)
sys.modules["src.backends.camel_integration"] = _fake_ci

import torch  # noqa: E402

from src.backends import steering_backend as sb  # noqa: E402


class BuildModelKwargsTests(unittest.TestCase):
    def test_cuda_unquantized_sets_dtype_and_device_map(self) -> None:
        kwargs = sb._build_model_kwargs(
            resolved_device="cuda",
            torch_dtype=torch.float16,
            is_quantized=False,
            trust_remote_code=False,
        )
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertEqual(kwargs["torch_dtype"], torch.float16)

    def test_cuda_quantized_omits_dtype(self) -> None:
        kwargs = sb._build_model_kwargs(
            resolved_device="cuda",
            torch_dtype=torch.float16,
            is_quantized=True,
            trust_remote_code=False,
        )
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertNotIn("torch_dtype", kwargs)

    def test_cpu_unquantized_uses_float32(self) -> None:
        kwargs = sb._build_model_kwargs(
            resolved_device="cpu",
            torch_dtype=torch.float16,
            is_quantized=False,
            trust_remote_code=True,
        )
        self.assertEqual(kwargs["torch_dtype"], torch.float32)
        self.assertTrue(kwargs["trust_remote_code"])
        self.assertNotIn("device_map", kwargs)

    def test_cpu_quantized_is_unsupported(self) -> None:
        with self.assertRaises(RuntimeError):
            sb._build_model_kwargs(
                resolved_device="cpu",
                torch_dtype=torch.float32,
                is_quantized=True,
                trust_remote_code=False,
            )


if __name__ == "__main__":
    unittest.main()
