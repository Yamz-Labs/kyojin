"""DFlash2 drafter of a Llama-architecture target (Nanbeige4.3-33B): the config reads the way the engine needs (CPU, header only).

Needs the drafter directory (NANB_DFLASH_DIR, default ~/models/drafters/nanb-dflash2-exl3br); skipped without it.
"""
import os
import unittest
from pathlib import Path

DIR = Path(os.environ.get("NANB_DFLASH_DIR", "~/models/drafters/nanb-dflash2-exl3br")).expanduser()


@unittest.skipUnless((DIR / "model.safetensors").exists(), "drafter directory not present")
class NanbDflashConfigTests(unittest.TestCase):
    def test_config(self):
        from exllamav3 import Config
        from exllamav3.architecture.dflash2 import DFlash2Model
        c = Config.from_directory(str(DIR))
        try:
            self.assertIs(c.model_classes["text"], DFlash2Model)
            self.assertEqual((c.block_size, c.tap_shift, c.mask_token_id), (8, 0, 0))
            self.assertEqual(list(c.target_layer_ids), [1, 25, 49])      # outputs of layers of the 52-layer target
            self.assertEqual((c.selector_top_k, c.selector_rank, c.conv_kernel_size, c.conv_group_size), (32, 128, 2, 16))
            self.assertTrue(c.bidirectional_block)            # is_causal false: the block sees itself both ways
        finally:
            c.stc.close()


if __name__ == "__main__":
    unittest.main()
