import pytest
from pathlib import Path

import config_helper as ch


# ---------------------------------------------------------------------------
# Small YAML writers
# ---------------------------------------------------------------------------

def _write_dicing_metadata_yaml(
    path,
    base_dir="/tmp/0deg",
    cal_file_path="/tmp/SpindleCal.txt",
    cutparams_filepath="/tmp/cutparams.txt",
    lens_metrology_file_path="/tmp/Lens_Met_0deg.dat",
    flange_metrology_file_path="/tmp/Flange_Met_0deg.dat",
    spindle="SpindleB",
    cuttype="Thick",
    blade_diameter=81.4,
    orientation="0deg",
):
    path = Path(path)
    path.write_text(
        f"paths:\n"
        f"  cal_file_path: '{cal_file_path}'\n"
        f"  cutparams_filepath: '{cutparams_filepath}'\n"
        f"\n"
        f"spindles:\n"
        f"  {spindle}:\n"
        f"    type: '{cuttype}'\n"
        f"    blade_diameter: {blade_diameter}\n"
        f"\n"
        f"orientations:\n"
        f"  {orientation}:\n"
        f"    lens_metrology_file_path: '{lens_metrology_file_path}'\n"
        f"    flange_metrology_file_path: '{flange_metrology_file_path}'\n"
        f"    base_dir: '{base_dir}'\n"
    )
    return path


def _write_testtouch_yaml(
    path,
    spindle="SpindleB",
    orientation="0deg",
    x_center_shift="+0.250",
    x_postcal_shift="+0.125",
    y_postcal_shift="-0.050",
    zcorr="-0.125",
):
    path = Path(path)
    path.write_text(
        f"{spindle}:\n"
        f"  x_center_shift: '{x_center_shift}'\n"
        f"  x_postcal_shift: '{x_postcal_shift}'\n"
        f"  y_postcal_shift: '{y_postcal_shift}'\n"
        f"  zcorr:\n"
        f"    {orientation}: '{zcorr}'\n"
        f"    90deg: '0'\n"
    )
    return path


def _write_lensparams_yaml(
    path,
    R=-818.9215895,
    k=-30.00046667,
    a1=-1.56460553e-4,
    a2=-3.06349779e-9,
    a3=5.00246955e-14,
    a4=0.0,
    t_ctr=53.507,
    diam=448.0,
    x_rot_shift=0.0,
    step_height=7.046,
    cut_diam=459.0,
):
    path = Path(path)
    path.write_text(
        "lensparams:\n"
        f"  R: {R}\n"
        f"  k: {k}\n"
        f"  a1: {a1}\n"
        f"  a2: {a2}\n"
        f"  a3: {a3}\n"
        f"  a4: {a4}\n"
        f"  t_ctr: {t_ctr}\n"
        f"  diam: {diam}\n"
        "\n"
        "alignment:\n"
        f"  x_rot_shift: {x_rot_shift}\n"
        "\n"
        "cut:\n"
        f"  step_height: {step_height}\n"
        f"  cut_diam: {cut_diam}\n"
    )
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_files(tmp_path):
    dicing_metadata_path = _write_dicing_metadata_yaml(
        tmp_path / "dicing_path_metadata.yaml"
    )
    testtouch_config_path = _write_testtouch_yaml(
        tmp_path / "lens_testtouches.yaml"
    )
    lensparams_config_path = _write_lensparams_yaml(
        tmp_path / "lensparams.yaml"
    )

    return {
        "dicing_metadata_path": dicing_metadata_path,
        "testtouch_config_path": testtouch_config_path,
        "lensparams_config_path": lensparams_config_path,
    }


@pytest.fixture
def loaded_configs(config_files):
    dicing_metadata = ch.load_yaml_config(config_files["dicing_metadata_path"])
    testtouch_cfg = ch.load_yaml_config(config_files["testtouch_config_path"])
    lensparams_cfg = ch.load_yaml_config(config_files["lensparams_config_path"])

    return {
        "dicing_metadata": dicing_metadata,
        "testtouch_cfg": testtouch_cfg,
        "lensparams_cfg": lensparams_cfg,
    }


# ---------------------------------------------------------------------------
# parse_signed_value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("+0.125", 0.125),
    ("-0.02834", -0.02834),
    ("0", 0.0),
    (0.0, 0.0),
    (5, 5.0),
])
def test_parse_signed_value_handles_plus_minus_and_numeric(value, expected):
    assert ch.parse_signed_value(value) == expected


def test_parse_signed_value_rejects_empty_string():
    with pytest.raises(ValueError):
        ch.parse_signed_value("")


def test_parse_signed_value_rejects_invalid_type():
    with pytest.raises(TypeError):
        ch.parse_signed_value(["0.1"])


# ---------------------------------------------------------------------------
# load_yaml_config
# ---------------------------------------------------------------------------

def test_load_yaml_config_returns_dict(config_files):
    cfg = ch.load_yaml_config(config_files["dicing_metadata_path"])
    assert isinstance(cfg, dict)
    assert "paths" in cfg
    assert "spindles" in cfg
    assert "orientations" in cfg


def test_load_yaml_config_raises_for_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError):
        ch.load_yaml_config(missing)


# ---------------------------------------------------------------------------
# get_shared_paths
# ---------------------------------------------------------------------------

def test_get_shared_paths_returns_expected_values(loaded_configs):
    paths = ch.get_shared_paths(loaded_configs["dicing_metadata"])

    assert paths["cal_file_path"] == "/tmp/SpindleCal.txt"
    assert paths["cutparams_filepath"] == "/tmp/cutparams.txt"


# ---------------------------------------------------------------------------
# get_spindle_settings
# ---------------------------------------------------------------------------

def test_get_spindle_settings_returns_expected_values(loaded_configs):
    spindle_settings = ch.get_spindle_settings(
        "SpindleB",
        loaded_configs["dicing_metadata"],
        loaded_configs["testtouch_cfg"],
    )

    assert spindle_settings["spindle"] == "SpindleB"
    assert spindle_settings["type"] == "Thick"
    assert spindle_settings["blade_diameter"] == 81.4
    assert spindle_settings["x_center_shift"] == 0.250
    assert spindle_settings["x_postcal_shift"] == 0.125
    assert spindle_settings["y_postcal_shift"] == -0.050
    assert spindle_settings["zcorr_by_orientation"] == {
        "0deg": "-0.125",
        "90deg": "0",
    }


def test_get_spindle_settings_raises_for_unknown_spindle(loaded_configs):
    with pytest.raises(KeyError):
        ch.get_spindle_settings(
            "S999",
            loaded_configs["dicing_metadata"],
            loaded_configs["testtouch_cfg"],
        )


# ---------------------------------------------------------------------------
# get_orientation_settings
# ---------------------------------------------------------------------------

def test_get_orientation_settings_returns_expected_values(loaded_configs):
    orientation_settings = ch.get_orientation_settings(
        "SpindleB",
        "0deg",
        loaded_configs["dicing_metadata"],
        loaded_configs["testtouch_cfg"],
    )

    assert orientation_settings["orientation"] == "0deg"
    assert orientation_settings["lens_metrology_file_path"] == "/tmp/Lens_Met_0deg.dat"
    assert orientation_settings["flange_metrology_file_path"] == "/tmp/Flange_Met_0deg.dat"
    assert str(orientation_settings["base_dir"]) == "/tmp/0deg"
    assert orientation_settings["zcorr"] == -0.125


def test_get_orientation_settings_raises_for_unknown_orientation(loaded_configs):
    with pytest.raises(KeyError):
        ch.get_orientation_settings(
            "SpindleB",
            "360deg",
            loaded_configs["dicing_metadata"],
            loaded_configs["testtouch_cfg"],
        )


# ---------------------------------------------------------------------------
# get_lensparams_settings
# ---------------------------------------------------------------------------

def test_get_lensparams_settings_returns_expected_values(loaded_configs):
    lensparams_settings = ch.get_lensparams_settings(loaded_configs["lensparams_cfg"])

    assert lensparams_settings["lensparams"]["R"] == -818.9215895
    assert lensparams_settings["lensparams"]["k"] == -30.00046667
    assert lensparams_settings["lensparams"]["diam"] == 448.0
    assert lensparams_settings["x_rot_shift"] == 0.0
    assert lensparams_settings["step_height"] == 7.046
    assert lensparams_settings["cut_diam"] == 459.0


# ---------------------------------------------------------------------------
# build_cut_output_paths
# ---------------------------------------------------------------------------

def test_build_cut_output_paths_returns_expected_paths():
    paths = ch.build_cut_output_paths(
        base_dir="/tmp/0deg",
        spindle="SpindleB",
        ftype="Thick",
    )

    assert str(paths["base_dir"]) == "/tmp/0deg"
    assert str(paths["spindle_dir"]) == "/tmp/0deg/SpindleB"
    assert str(paths["noshift_dir"]) == "/tmp/0deg/SpindleB/CutCammingThick-Noshift"
    assert str(paths["shifted_dir"]) == "/tmp/0deg/SpindleB/CutCammingThick"
    assert str(paths["noshift_master"]) == "/tmp/0deg/SpindleB/CutCammingThick-Noshift/Master.txt"
    assert str(paths["shifted_master"]) == "/tmp/0deg/SpindleB/CutCammingThick/Master.txt"
    assert paths["cam_prefix"] == "CutCamThick"


# ---------------------------------------------------------------------------
# get_cut_context
# ---------------------------------------------------------------------------

def test_get_cut_context_returns_expected_merged_context(config_files):
    context = ch.get_cut_context(
        spindle="SpindleB",
        orientation="0deg",
        dicing_metadata_path=config_files["dicing_metadata_path"],
        testtouch_config_path=config_files["testtouch_config_path"],
        lensparams_config_path=config_files["lensparams_config_path"],
    )

    assert context["cal_file_path"] == "/tmp/SpindleCal.txt"
    assert context["cutparams_filepath"] == "/tmp/cutparams.txt"
    assert context["spindle"] == "SpindleB"
    assert context["type"] == "Thick"
    assert context["blade_diameter"] == 81.4
    assert context["x_center_shift"] == 0.250
    assert context["x_postcal_shift"] == 0.125
    assert context["y_postcal_shift"] == -0.050
    assert context["orientation"] == "0deg"
    assert context["lens_metrology_file_path"] == "/tmp/Lens_Met_0deg.dat"
    assert context["flange_metrology_file_path"] == "/tmp/Flange_Met_0deg.dat"
    assert str(context["base_dir"]) == "/tmp/0deg"
    assert context["zcorr"] == -0.125
    assert context["x_total_shift"] == 0.375
    assert context["x_rot_shift"] == 0.0
    assert context["step_height"] == 7.046
    assert context["cut_diam"] == 459.0
    assert context["cam_prefix"] == "CutCamThick"


def test_get_cut_context_without_lensparams_still_works(config_files):
    context = ch.get_cut_context(
        spindle="SpindleB",
        orientation="0deg",
        dicing_metadata_path=config_files["dicing_metadata_path"],
        testtouch_config_path=config_files["testtouch_config_path"],
        lensparams_config_path=None,
    )

    assert context["spindle"] == "SpindleB"
    assert context["orientation"] == "0deg"
    assert context["x_total_shift"] == 0.375
    assert "lensparams" not in context
    assert "x_rot_shift" not in context
    assert "step_height" not in context
    assert "cut_diam" not in context
