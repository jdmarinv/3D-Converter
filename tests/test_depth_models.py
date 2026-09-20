import unittest

from src.depth_models import DEPTH_MODELS, DEPTH_MODEL_BY_KEY, get_depth_model_spec


class DepthModelRegistryTests(unittest.TestCase):
    def test_all_pro_models_are_registered_once(self):
        self.assertEqual(len(DEPTH_MODELS), 21)  # built-in + 20 Pro models
        self.assertEqual(len(DEPTH_MODEL_BY_KEY), len(DEPTH_MODELS))
        self.assertEqual(
            {model.backend for model in DEPTH_MODELS},
            {"builtin", "onnx", "hf", "da3", "zoe", "prompt_hf"},
        )

    def test_metric_models_are_identified_for_inverse_depth(self):
        for key in ("zoedepth-nk", "da3metric-large", "da-v2-metric-outdoor-large", "prompt-da-vits-transparent"):
            self.assertTrue(get_depth_model_spec(key).metric, key)

    def test_unknown_model_is_rejected_before_conversion(self):
        with self.assertRaisesRegex(ValueError, "Unknown depth model"):
            get_depth_model_spec("does-not-exist")


if __name__ == "__main__":
    unittest.main()
