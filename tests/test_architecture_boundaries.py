import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_engine_does_not_dispatch_on_backend_or_spec_names(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "flashengine" / "engine").glob("*.py")
        )
        for fragment in (
            "spec_method ==",
            "spec_method in",
            "spec_method not in",
            "backend.name ==",
            "backend ==",
        ):
            self.assertNotIn(fragment, sources)

    def test_scripts_use_observation_api_instead_of_monkey_patches(self):
        sources = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (ROOT / "scripts").glob("*.py")
        )
        self.assertNotIn(".schedule =", sources)
        self.assertNotIn(".spec_round =", sources)
        self.assertNotIn("model_runner.", sources)


if __name__ == "__main__":
    unittest.main()
