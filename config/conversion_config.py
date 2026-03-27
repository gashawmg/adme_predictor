# config/conversion_config.py
"""Conversion configuration for log to actual values."""

# Conversion configuration
# Format: log_name -> {display_name, unit, log_scale, multiplier, description,
#                      reference_range, challenge_name}
CONVERSION_CONFIG = {
    "LogD": {
        "display_name": "LogD",
        "unit": "",
        "log_scale": False,
        "multiplier": 1,
        "description": (
            "Lipophilicity at physiological pH (7.4). Controls membrane permeability, "
            "protein binding, and metabolic stability. Optimal range for orally "
            "bioavailable drugs is -1 to +4."
        ),
        "reference_range": {
            "low": -1,
            "high": 4,
            "preferred_direction": "neutral",
            "note": "Optimal -1 to +4; > 5 indicates poor aqueous solubility risk",
        },
        "challenge_name": "LogD",
    },
    "LogS": {
        "display_name": "KSol",
        "unit": "µM",
        "log_scale": True,
        "multiplier": 1e-6,
        "description": (
            "Kinetic aqueous solubility. Drives oral bioavailability and formulation "
            "complexity. Values below 10 µM are generally considered a liability "
            "for oral drugs."
        ),
        "reference_range": {
            "low": 10,
            "high": None,
            "preferred_direction": "higher",
            "note": "> 100 µM excellent; 10–100 µM acceptable; < 1 µM liability",
        },
        "challenge_name": "Solubility_kin",
    },
    "Log_HLM_CLint": {
        "display_name": "HLM CLint",
        "unit": "mL/min/kg",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Intrinsic clearance in human liver microsomes. Predicts hepatic metabolic "
            "stability and half-life in humans. Low CLint (< 30 mL/min/kg) is preferred "
            "for oral drugs requiring systemic exposure."
        ),
        "reference_range": {
            "low": None,
            "high": 30,
            "preferred_direction": "lower",
            "note": "< 30 low clearance; 30–70 moderate; > 70 high clearance",
        },
        "challenge_name": "HLM_CLint",
    },
    "Log_MLM_CLint": {
        "display_name": "MLM CLint",
        "unit": "mL/min/kg",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Intrinsic clearance in mouse liver microsomes. Used to predict rodent PK "
            "and bridge to human clearance estimates in early DMPK triage."
        ),
        "reference_range": {
            "low": None,
            "high": 30,
            "preferred_direction": "lower",
            "note": "< 30 low clearance; 30–70 moderate; > 70 high clearance",
        },
        "challenge_name": "MLM_CLint",
    },
    "Log_Caco_Papp_AB": {
        "display_name": "Caco-2 Papp A>B",
        "unit": "10⁻⁶ cm/s",
        "log_scale": True,
        "multiplier": 1e-6,
        "description": (
            "Apparent permeability from apical to basolateral in Caco-2 monolayers. "
            "Surrogate for intestinal absorption. Papp > 10 × 10⁻⁶ cm/s is considered "
            "high permeability."
        ),
        "reference_range": {
            "low": 10,
            "high": None,
            "preferred_direction": "higher",
            "note": "> 10 high permeability; 1–10 moderate; < 1 low permeability",
        },
        "challenge_name": "Caco2_Papp_AB",
    },
    "Log_Caco_ER": {
        "display_name": "Caco-2 Efflux",
        "unit": "",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Efflux ratio = Papp B→A / Papp A→B. Values > 2 suggest active efflux "
            "(e.g., P-glycoprotein), which can limit oral absorption and CNS penetration."
        ),
        "reference_range": {
            "low": None,
            "high": 2,
            "preferred_direction": "lower",
            "note": "< 2 no significant efflux; > 2 potential P-gp substrate",
        },
        "challenge_name": "Caco2_ER",
    },
    "Log_Mouse_PPB": {
        "display_name": "MPPB",
        "unit": "% Unbound",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Fraction of drug unbound in mouse plasma. High protein binding (< 1% free) "
            "limits the free drug available for pharmacological activity and clearance."
        ),
        "reference_range": {
            "low": 1.0,
            "high": None,
            "preferred_direction": "higher",
            "note": "> 1% free fraction preferred; < 0.1% highly bound",
        },
        "challenge_name": "Mouse_PPB",
    },
    "Log_Mouse_BPB": {
        "display_name": "MBPB",
        "unit": "% Unbound",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Fraction unbound in mouse brain tissue. Combined with plasma binding, "
            "used to estimate CNS drug exposure and Kp,uu,brain."
        ),
        "reference_range": {
            "low": 1.0,
            "high": None,
            "preferred_direction": "higher",
            "note": "> 1% free fraction preferred for CNS penetration estimates",
        },
        "challenge_name": "Mouse_BPB",
    },
    "Log_Mouse_MPB": {
        "display_name": "MGMB",
        "unit": "% Unbound",
        "log_scale": True,
        "multiplier": 1,
        "description": (
            "Fraction unbound in mouse gastrocnemius muscle. Used for volume of "
            "distribution estimation and tissue distribution modeling."
        ),
        "reference_range": {
            "low": 1.0,
            "high": None,
            "preferred_direction": "higher",
            "note": "> 1% free fraction preferred; used in tissue distribution models",
        },
        "challenge_name": "Mouse_MPB",
    },
}

# ---------------------------------------------------------------------------
# Property grouping by ADME category
# ---------------------------------------------------------------------------
PROPERTY_GROUPS = {
    "Physicochemical": ["LogD", "LogS"],
    "Absorption": ["Log_Caco_Papp_AB", "Log_Caco_ER"],
    "Distribution": ["Log_Mouse_PPB", "Log_Mouse_BPB", "Log_Mouse_MPB"],
    "Metabolism": ["Log_HLM_CLint", "Log_MLM_CLint"],
}

# Reverse mapping: display_name -> log_name
DISPLAY_TO_LOG_NAME = {v["display_name"]: k for k, v in CONVERSION_CONFIG.items()}

# Log name to display name
LOG_TO_DISPLAY_NAME = {k: v["display_name"] for k, v in CONVERSION_CONFIG.items()}


def convert_log_to_actual(log_value: float, log_name: str) -> float:
    """
    Convert a log-scale value to actual value.

    Formula: actual = (10^log_value) / multiplier
    """
    if log_name not in CONVERSION_CONFIG:
        return log_value
    config = CONVERSION_CONFIG[log_name]
    if not config["log_scale"]:
        return log_value
    return (10**log_value) / config["multiplier"]


def convert_actual_to_log(actual_value: float, log_name: str) -> float:
    """Convert an actual value back to log-scale."""
    import numpy as np

    if log_name not in CONVERSION_CONFIG:
        return actual_value
    config = CONVERSION_CONFIG[log_name]
    if not config["log_scale"]:
        return actual_value
    return np.log10(actual_value * config["multiplier"])


def get_unit(log_name: str) -> str:
    """Get the unit for a property."""
    if log_name in CONVERSION_CONFIG:
        return CONVERSION_CONFIG[log_name]["unit"]
    return ""


def get_display_name(log_name: str) -> str:
    """Get the display name for a property."""
    if log_name in CONVERSION_CONFIG:
        return CONVERSION_CONFIG[log_name]["display_name"]
    return log_name


def format_value_with_unit(value: float, log_name: str, decimals: int = 3) -> str:
    """Format a value with its unit."""
    unit = get_unit(log_name)
    if unit:
        return f"{value:.{decimals}f} {unit}"
    return f"{value:.{decimals}f}"


def is_in_drug_like_range(value: float, log_name: str) -> bool | None:
    """Return True if value is within the drug-like reference range, False if outside, None if no range defined."""
    if log_name not in CONVERSION_CONFIG:
        return None
    rng = CONVERSION_CONFIG[log_name].get("reference_range", {})
    low = rng.get("low")
    high = rng.get("high")
    if low is None and high is None:
        return None
    if low is not None and value < low:
        return False
    if high is not None and value > high:
        return False
    return True
