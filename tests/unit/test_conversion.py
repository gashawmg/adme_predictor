"""Unit tests for conversion_config — no model files required."""

import math
import pytest
from config.conversion_config import (
    CONVERSION_CONFIG,
    convert_log_to_actual,
    get_display_name,
    get_unit,
)

ALL_TARGETS = [
    "LogD", "LogS", "Log_HLM_CLint", "Log_MLM_CLint",
    "Log_Caco_Papp_AB", "Log_Caco_ER",
    "Log_Mouse_PPB", "Log_Mouse_BPB", "Log_Mouse_MPB",
]


class TestConversionConfig:
    def test_all_targets_present(self):
        for t in ALL_TARGETS:
            assert t in CONVERSION_CONFIG, f"{t} missing from CONVERSION_CONFIG"

    def test_required_keys_per_target(self):
        required = {"display_name", "unit", "log_scale", "multiplier", "description"}
        for t, cfg in CONVERSION_CONFIG.items():
            assert required.issubset(cfg.keys()), f"{t} missing keys: {required - cfg.keys()}"

    def test_logd_is_not_log_scale(self):
        assert CONVERSION_CONFIG["LogD"]["log_scale"] is False

    def test_logs_is_log_scale(self):
        assert CONVERSION_CONFIG["LogS"]["log_scale"] is True

    def test_multiplier_is_positive(self):
        for t, cfg in CONVERSION_CONFIG.items():
            assert cfg["multiplier"] > 0, f"{t} has non-positive multiplier"


class TestConvertLogToActual:
    def test_log_scale_zero_returns_inverse_multiplier(self):
        # LogS: log_value=0 → 10^0 / 1e-6 = 1e6 µM
        result = convert_log_to_actual(0.0, "LogS")
        assert math.isclose(result, 1e6, rel_tol=1e-9)

    def test_non_log_scale_passthrough(self):
        # LogD is not log-scaled — value returned as-is
        result = convert_log_to_actual(2.5, "LogD")
        assert result == 2.5

    def test_unknown_target_passthrough(self):
        result = convert_log_to_actual(3.14, "SomeUnknownTarget")
        assert result == 3.14

    def test_hlm_clint_log1_returns_10(self):
        # Log_HLM_CLint: multiplier=1, log_value=1 → 10^1 / 1 = 10
        result = convert_log_to_actual(1.0, "Log_HLM_CLint")
        assert math.isclose(result, 10.0, rel_tol=1e-9)

    def test_caco_papp_log_value(self):
        # Log_Caco_Papp_AB: multiplier=1e-6, log_value=0 → 10^0 / 1e-6 = 1e6
        result = convert_log_to_actual(0.0, "Log_Caco_Papp_AB")
        assert math.isclose(result, 1e6, rel_tol=1e-9)


class TestGetters:
    def test_get_unit_logd_empty(self):
        assert get_unit("LogD") == ""

    def test_get_unit_logs_micromolar(self):
        assert get_unit("LogS") == "µM"

    def test_get_unit_unknown_returns_empty(self):
        assert get_unit("NonExistent") == ""

    def test_get_display_name_logd(self):
        assert get_display_name("LogD") == "LogD"

    def test_get_display_name_hlm(self):
        assert get_display_name("Log_HLM_CLint") == "HLM CLint"

    def test_get_display_name_unknown_returns_key(self):
        assert get_display_name("UnknownKey") == "UnknownKey"
