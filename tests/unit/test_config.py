"""Unit tests for model and conversion config integrity — no model files required."""

from config.conversion_config import CONVERSION_CONFIG
from config.model_config import ALL_TARGETS, MULTITASK_CONFIG, TARGET_CONFIG

ALL_EXPECTED_TARGETS = [
    "LogD",
    "LogS",
    "Log_HLM_CLint",
    "Log_MLM_CLint",
    "Log_Caco_Papp_AB",
    "Log_Caco_ER",
    "Log_Mouse_PPB",
    "Log_Mouse_BPB",
    "Log_Mouse_MPB",
]


class TestAllTargets:
    def test_all_expected_targets_in_all_targets(self):
        for t in ALL_EXPECTED_TARGETS:
            assert t in ALL_TARGETS, f"{t} missing from ALL_TARGETS"

    def test_no_duplicate_targets(self):
        assert len(ALL_TARGETS) == len(set(ALL_TARGETS))


class TestTargetConfig:
    def test_every_target_has_config(self):
        for t in ALL_TARGETS:
            assert t in TARGET_CONFIG, f"{t} missing from TARGET_CONFIG"

    def test_every_config_has_required_keys(self):
        for t, cfg in TARGET_CONFIG.items():
            assert "strategy" in cfg, f"{t} missing 'strategy'"
            assert "training_style" in cfg, f"{t} missing 'training_style'"

    def test_multitask_targets_reference_valid_group(self):
        for t, cfg in TARGET_CONFIG.items():
            if cfg.get("is_multitask"):
                group = cfg.get("multitask_group")
                assert group in MULTITASK_CONFIG, (
                    f"{t} references unknown multitask_group '{group}'"
                )

    def test_multitask_config_groups_have_required_keys(self):
        required = {"targets", "model_folder", "training_style", "scaler_type"}
        for group, cfg in MULTITASK_CONFIG.items():
            assert required.issubset(cfg.keys()), f"{group} missing keys: {required - cfg.keys()}"


class TestConversionConfigCompleteness:
    def test_all_targets_have_conversion_entry(self):
        for t in ALL_TARGETS:
            assert t in CONVERSION_CONFIG, f"{t} missing from CONVERSION_CONFIG"

    def test_target_config_and_conversion_config_in_sync(self):
        """Every target in TARGET_CONFIG must have a CONVERSION_CONFIG entry."""
        for t in TARGET_CONFIG:
            assert t in CONVERSION_CONFIG, f"TARGET_CONFIG has '{t}' but CONVERSION_CONFIG does not"
