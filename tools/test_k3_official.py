from __future__ import annotations

import pathlib
import tempfile
import unittest
from unittest import mock

from tools import k3_official


class OfficialBootstrapTests(unittest.TestCase):
    def test_legacy_metadata_path_is_unchanged(self):
        with mock.patch.dict(
            "os.environ",
            {
                "K3_EXPERT_SOURCE": "cache-http",
                "K3_RESIDENT_SOURCE": "cache-http",
            },
            clear=True,
        ):
            self.assertFalse(k3_official.direct_local_requested())
            self.assertEqual(
                k3_official.metadata_dir("/deltafin"),
                pathlib.Path("/deltafin/k3-meta"),
            )

    def test_direct_metadata_requires_complete_official_directory(self):
        required = (
            "config.json",
            "model.safetensors.index.json",
            "modeling_kimi_linear.py",
            "configuration_kimi_k3.py",
            "tokenization_kimi.py",
            "tiktoken.model",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            for name in required:
                (root / name).touch()
            with mock.patch.dict(
                "os.environ",
                {
                    "K3_EXPERT_SOURCE": "direct-shards",
                    "K3_MODEL_DIR": str(root),
                },
                clear=True,
            ):
                self.assertTrue(k3_official.direct_local_requested())
                self.assertEqual(k3_official.metadata_dir("/unused"), root)
                (root / "tokenization_kimi.py").unlink()
                with self.assertRaisesRegex(RuntimeError, "official files"):
                    k3_official.model_dir()


if __name__ == "__main__":
    unittest.main()
