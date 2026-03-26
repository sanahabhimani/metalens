import numpy as np
from pathlib import Path
from unittest.mock import patch

import housekeeping
import lensfit


# ---------------------------------------------------------------------------
# Small file writers
# ---------------------------------------------------------------------------

def _write_cutparams_file(path, thick_depth=1.0, med_depth=0.5, thin_depth=0.25, cutpitch=4.0):
    """
    Write a cut params file in the format expected by core_utils.get_cut_parameters.
    """
    path = Path(path)
    path.write_text(
        "value\n"
        f"{thick_depth}\n"
        f"{med_depth}\n"
        f"{thin_depth}\n"
        f"{cutpitch}\n"
    )
    return path


def _write_spindlecal_file(path, spindle="S1", xoffset=1.0, yoffset=2.0, zoffset=3.0):
    """
    Write a spindle calibration file in the format expected by core_utils.get_spindle_offsets.
    """
    path = Path(path)
    path.write_text(
        "Spindle Xoffset Yoffset Zoffset\n"
        "S1 1.0 2.0 3.0\n"
        "S2 4.0 5.0 6.0\n"
    )
    return path


def _write_dicing_metadata_yaml(
    path,
    base_dir,
    cal_file_path,
    cutparams_filepath,
    metrology_file_path,
    spindle="S1",
    cuttype="Thick",
    blade_diameter=100.0,
    orientation="Face1",
):
    """
    Write a minimal dicing metadata YAML file compatible with get_cut_context.
    """
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
        f"    metrology_file_path: '{metrology_file_path}'\n"
        f"    base_dir: '{base_dir}'\n"
    )
    return path


def _write_testtouch_yaml(
    path,
    spindle="S1",
    orientation="Face1",
    x_center_shift="+0.250",
    x_postcal_shift="+0.125",
    y_postcal_shift="0",
    zcorr="-0.125",
):
    """
    Write a minimal test-touch config YAML.
    """
    path = Path(path)
    path.write_text(
        f"{spindle}:\n"
        f"  x_center_shift: '{x_center_shift}'\n"
        f"  x_postcal_shift: '{x_postcal_shift}'\n"
        f"  y_postcal_shift: '{y_postcal_shift}'\n"
        f"  zcorr:\n"
        f"    {orientation}: '{zcorr}'\n"
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
    cut_diam=8.0,
):
    """
    Write a minimal lensparams.yaml compatible with your helper layer.
    """
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

def _compare_file_contents(file1, file2):
    """
    Compare two files but ignore lines that contain absolute paths (e.g., ;Filename).
    """
    lines1 = Path(file1).read_text().splitlines()
    lines2 = Path(file2).read_text().splitlines()

    # Remove filename lines
    lines1 = [l for l in lines1 if not l.startswith(";Filename")]
    lines2 = [l for l in lines2 if not l.startswith(";Filename")]

    assert lines1 == lines2

def _compare_dir_trees(dir1, dir2):
    """
    Compare two directory trees by relative file list and file contents.
    """
    dir1 = Path(dir1)
    dir2 = Path(dir2)

    files1 = sorted([p.relative_to(dir1) for p in dir1.rglob("*") if p.is_file()])
    files2 = sorted([p.relative_to(dir2) for p in dir2.rglob("*") if p.is_file()])

    assert files1 == files2

    for rel in files1:
        _compare_file_contents(dir1 / rel, dir2 / rel)


# ---------------------------------------------------------------------------
# HK integration test
# ---------------------------------------------------------------------------

def test_build_cut_directories_hk_config_matches_legacy(tmp_path):
    """
    Run the housekeeping pipeline both ways:
      1. direct legacy call
      2. config-driven wrapper

    Then confirm the cut directories and generated files match exactly.
    """
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    cal_file = _write_spindlecal_file(inputs_dir / "SpindleCal.txt")
    cutparams_file = _write_cutparams_file(inputs_dir / "cutparams.txt", cutpitch=4.0)

    legacy_base = tmp_path / "legacy_hk" / "Face1"
    config_base = tmp_path / "config_hk" / "Face1"
    legacy_base.mkdir(parents=True)
    config_base.mkdir(parents=True)

    dicing_yaml = _write_dicing_metadata_yaml(
        tmp_path / "dicing_path_metadata_hk.yaml",
        base_dir=config_base,
        cal_file_path=cal_file,
        cutparams_filepath=cutparams_file,
        metrology_file_path=tmp_path / "dummy_metrology.dat",
        spindle="S1",
        cuttype="Thick",
        blade_diameter=100.0,
        orientation="Face1",
    )

    testtouch_yaml = _write_testtouch_yaml(
        tmp_path / "lens_testtouches_hk.yaml",
        spindle="S1",
        orientation="Face1",
    )

    p = [0.1, 0.2, 0.3]

    # Legacy call
    housekeeping.generate_hkcut_files(
        p=p,
        pathname=str(legacy_base) + "/",
        spindle="S1",
        calibrationfilepath=str(cal_file),
        cutparamsfile=str(cutparams_file),
        bladeradius=50.0,
        cuttype="Thick",
        xstart=0.0,
        xend=12.0,
        ystart=0.0,
        yend=2.0,
        use_noshift_suffix=True,
    )

    # Config-driven call
    housekeeping.generate_hkcut_files_config(
        p=p,
        spindle="S1",
        orientation="Face1",
        dicing_metadata_path=str(dicing_yaml),
        testtouch_config_path=str(testtouch_yaml),
        xstart=0.0,
        xend=12.0,
        ystart=0.0,
        yend=2.0,
        use_noshift_suffix=True,
    )

    legacy_spindle_dir = legacy_base / "S1"
    config_spindle_dir = config_base / "S1"

    assert legacy_spindle_dir.exists()
    assert config_spindle_dir.exists()

    # Housekeeping creates all 3 type folders, even if only one is written into.
    for folder_name in ["CutCammingThick-Noshift", "CutCammingMed-Noshift", "CutCammingThin-Noshift"]:
        assert (legacy_spindle_dir / folder_name).exists()
        assert (config_spindle_dir / folder_name).exists()

    _compare_dir_trees(legacy_spindle_dir, config_spindle_dir)


# ---------------------------------------------------------------------------
# Lens integration test
# ---------------------------------------------------------------------------

def test_build_cut_directories_lens_config_matches_legacy(tmp_path):
    """
    Run the lens cut pipeline both ways:
      1. direct legacy call
      2. config-driven wrapper

    Patch the heavy lens-surface math to deterministic simple functions so the
    actual file-writing path can be exercised and compared exactly.
    """
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    cal_file = _write_spindlecal_file(inputs_dir / "SpindleCal.txt")
    cutparams_file = _write_cutparams_file(inputs_dir / "cutparams.txt", cutpitch=2.0)

    legacy_base = tmp_path / "legacy_lens" / "Face1"
    config_base = tmp_path / "config_lens" / "Face1"
    legacy_base.mkdir(parents=True)
    config_base.mkdir(parents=True)

    metrology_file = inputs_dir / "Lens_Met_Face1.dat"
    metrology_file.write_text("0,0,0,0\n")

    dicing_yaml = _write_dicing_metadata_yaml(
        tmp_path / "dicing_path_metadata_lens.yaml",
        base_dir=config_base,
        cal_file_path=cal_file,
        cutparams_filepath=cutparams_file,
        metrology_file_path=metrology_file,
        spindle="S1",
        cuttype="Thick",
        blade_diameter=100.0,
        orientation="Face1",
    )

    testtouch_yaml = _write_testtouch_yaml(
        tmp_path / "lens_testtouches_lens.yaml",
        spindle="S1",
        orientation="Face1",
    )

    lensparams_yaml = _write_lensparams_yaml(
        tmp_path / "lensparams.yaml",
        cut_diam=8.0,
        step_height=7.046,
        x_rot_shift=0.0,
    )

    p = [1.0, 2.0, 3.0, 0.1, 0.2]
    p2 = [1.5, 2.5, 3.5, 0.1, 0.2]

    lensparams = [-818.9215895, -30.00046667, -1.56460553e-4, -3.06349779e-9,
                  5.00246955e-14, 0.0, 53.507, 448.0]

    def fake_gradF(p_in, xx, yy, lensparams_in):
        return 0.0, 0.0, 1.0

    def fake_FlrtNoball(p_in, xx, yy, lensparams_in):
        return float(xx), float(yy), float(0.5 * xx + 0.1 * yy)

    def fake_gradFplane(p_in, xx, yy, lensparams_in, stepheight_in):
        return 0.0, 0.0, 1.0

    def fake_PlaneNoball(p_in, xx, yy, lensparams_in, stepheight_in):
        return float(xx), float(yy), -1.0e6

    with patch("lensfit.gradF", side_effect=fake_gradF), \
         patch("lensfit.FlrtNoball", side_effect=fake_FlrtNoball), \
         patch("lensfit.gradFplane", side_effect=fake_gradFplane), \
         patch("lensfit.PlaneNoball", side_effect=fake_PlaneNoball):

        # Legacy call
        lensfit.generate_lens_cut_files(
            p=p,
            p2=p2,
            pathname=str(legacy_base) + "/",
            spindle="S1",
            calibrationfilepath=str(cal_file),
            cutparamsfile=str(cutparams_file),
            cutdiameter=8.0,
            bladeradius=50.0,
            cuttype="Thick",
            lensparams=lensparams,
            afixed=0.1,
            bfixed=0.2,
            stepheight=7.046,
            use_fit="p",
            x_rot_shift=0.0,
            yres=1.0,
        )

        # Config-driven call
        lensfit.generate_lens_cutfiles_fromconfig(
            p=p,
            p2=p2,
            spindle="S1",
            orientation="Face1",
            dicing_metadata_path=str(dicing_yaml),
            testtouch_config_path=str(testtouch_yaml),
            lensparams_config_path=str(lensparams_yaml),
            afixed=0.1,
            bfixed=0.2,
            use_fit="p",
            yres=1.0,
        )

    legacy_cut_dir = legacy_base / "S1" / "CutCammingThick-Noshift"
    config_cut_dir = config_base / "S1" / "CutCammingThick-Noshift"

    assert legacy_cut_dir.exists()
    assert config_cut_dir.exists()
    assert (legacy_cut_dir / "Master.txt").exists()
    assert (config_cut_dir / "Master.txt").exists()

    _compare_dir_trees(legacy_cut_dir, config_cut_dir)
