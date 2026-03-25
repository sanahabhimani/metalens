from pathlib import Path
import yaml

def load_yaml_config(config_path):
    """
    Load a YAML config file.

    Parameters
    ----------
    config_path : str
        Path to YAML config file

    Returns
    -------
    dict
        Parsed YAML contents
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {config_path}")

    if not isinstance(data, dict):
        raise TypeError(f"Expected dict at top level in {config_path}")

    return data


def parse_signed_value(value):
    """
    Convert string or numeric value to float.

    Parameters
    ----------
    value : str or float or int
        Value like '+0.010', '-0.02834', or numeric

    Returns
    -------
    float
    """
    if isinstance(value, (int, float)):
        return float(value)

    if not isinstance(value, str):
        raise TypeError(f"Invalid type for signed value: {type(value)}")

    stripped = value.strip()

    if stripped == "":
        raise ValueError("Empty string cannot be parsed as float")

    return float(stripped)


def get_spindle_settings(spindle, spindles_cfg, testtouch_cfg):
    """
    Resolve spindle-level settings.

    Parameters
    ----------
    spindle : str
    spindles_cfg : dict
        Contents of spindles.yaml
    testtouch_cfg : dict
        Contents of lens_testtouches.yaml

    Returns
    -------
    dict
    """
    spindle_block = spindles_cfg.get("spindles", {}).get(spindle)
    if spindle_block is None:
        raise KeyError(f"Spindle '{spindle}' not found in spindles config")

    testtouch_block = testtouch_cfg.get(spindle)
    if testtouch_block is None:
        raise KeyError(f"Spindle '{spindle}' not found in test-touch config")

    return {
        "spindle": spindle,
        "type": spindle_block["type"],
        "blade_diameter": spindle_block["blade_diameter"],
        "x_center_shift": parse_signed_value(testtouch_block['x_center_shift']),
        "x_postcal_shift": parse_signed_value(testtouch_block['x_postcal_shift']),
        "y_postcal_shift": parse_signed_value(testtouch_block['y_postcal_shift']),
        "zcorr_by_orientation": testtouch_block['zcorr'],
    }

def get_orientation_settings(spindle, orientation, spindles_cfg, testtouch_cfg):
    """
    Resolve orientation-level settings.

    Parameters
    ----------
    spindle : str
    orientation : str
    spindles_cfg : dict
    testtouch_cfg : dict

    Returns
    -------
    dict
    """
    orientation_block = spindles_cfg.get("orientations", {}).get(orientation)
    if orientation_block is None:
        raise KeyError(f"Orientation '{orientation}' not found")

    spindle_settings = get_spindle_settings(
        spindle, spindles_cfg, testtouch_cfg
    )

    zcorr = spindle_settings["zcorr_by_orientation"].get(orientation, "0")

    return {
        "orientation": orientation,
        "metrology_file_path": orientation_block["metrology_file_path"],
        "base_dir": orientation_block["base_dir"],
        "zcorr": parse_signed_value(zcorr),
    }


def get_shared_paths(spindles_cfg):
    """
    Resolve shared file paths.

    Parameters
    ----------
    spindles_cfg : dict

    Returns
    -------
    dict
    """
    paths_block = spindles_cfg.get("paths", {})

    if not paths_block:
        raise KeyError("Missing 'paths' block in spindles config")

    return {
        "cal_file_path": paths_block["cal_file_path"],
        "cutparams_filepath": paths_block["cutparams_filepath"],
    }

def get_lensparams_settings(lensparams_cfg):
    """
    Resolve lensparams config.

    Parameters
    ----------
    lensparams_cfg : dict

    Returns
    -------
    dict
    """
    lensparams_block = lensparams_cfg.get("lensparams", {})
    alignment_block = lensparams_cfg.get("alignment", {})
    cut_block = lensparams_cfg.get("cut", {})

    return {
        "lensparams": lensparams_block,
        "x_rot_shift": alignment_block.get("x_rot_shift", 0.0),
        "step_height": cut_block.get("step_height"),
        "cut_diam": cut_block.get("cut_diam"),
    }

def build_cut_output_paths(base_dir, spindle, ftype):
    """
    Build output directory structure.

    Parameters
    ----------
    base_dir : str
    spindle : str
    ftype : str

    Returns
    -------
    dict
    """
    base_dir = Path(base_dir)
    spindle_dir = base_dir / spindle

    noshift_dir = spindle_dir / f"CutCamming{ftype}-Noshift"
    shifted_dir = spindle_dir / f"CutCamming{ftype}"

    return {
        "base_dir": base_dir,
        "spindle_dir": spindle_dir,
        "noshift_dir": noshift_dir,
        "shifted_dir": shifted_dir,
        "noshift_master": noshift_dir / "Master.txt",
        "shifted_master": shifted_dir / "Master.txt",
        "cam_prefix": f"CutCam{ftype}",
    }


def get_cut_context(
    spindle,
    orientation,
    spindles_config_path,
    testtouch_config_path,
    lensparams_config_path=None,
):
    """
    Resolve full context for a spindle + orientation.

    Parameters
    ----------
    spindle : str
    orientation : str
    spindles_config_path : str
    testtouch_config_path : str
    lensparams_config_path : str or None

    Returns
    -------
    dict
    """
    spindles_cfg = load_yaml_config(spindles_config_path)
    testtouch_cfg = load_yaml_config(testtouch_config_path)

    shared = get_shared_paths(spindles_cfg)
    spindle_settings = get_spindle_settings(
        spindle, spindles_cfg, testtouch_cfg
    )
    orientation_settings = get_orientation_settings(
        spindle, orientation, spindles_cfg, testtouch_cfg
    )

    output_paths = build_cut_output_paths(
        base_dir=orientation_settings["base_dir"],
        spindle=spindle,
        ftype=spindle_settings["type"],
    )

    context = {
        **shared,
        **spindle_settings,
        **orientation_settings,
        **output_paths,
    }

    if lensparams_config_path is not None:
        lensparams_cfg = load_yaml_config(lensparams_config_path)
        context.update(get_lensparams_settings(lensparams_cfg))

    context["x_total_shift"] = (
        context["x_center_shift"] + context["x_postcal_shift"]
    )

    return context
