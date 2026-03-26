import numpy as np
from pathlib import Path

import housekeeping
import core_utils
import config_helper as ch


# ---------------------------------------------------------------------------
# Small file writers
# ---------------------------------------------------------------------------

def _write_cutparams_file(path, thick_depth=1.0, med_depth=0.5, thin_depth=0.25, cutpitch=4.0):
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


def _ensure_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


# ---------------------------------------------------------------------------
# Shift test
# ---------------------------------------------------------------------------

def test_shiftxz_nocomp_applies_expected_shifts_and_preserves_pitch(tmp_path):
    """
    Build a real HK no-shift directory, compute x_total_shift and zcorr from config,
    apply shiftXZ_nocomp, then verify:

      1. X in shifted Master.txt = original X + x_total_shift
      2. Z in shifted Master.txt = original Z + zcorr
      3. Z in each shifted .Cam file = original Z + zcorr
      4. Y columns in .Cam files remain unchanged
      5. X pitch in Master.txt matches the cut pitch from the cut params file
    """
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    cal_file = _write_spindlecal_file(inputs_dir / "SpindleCal.txt")
    cutparams_file = _write_cutparams_file(inputs_dir / "cutparams.txt", cutpitch=4.0)

    base_dir = tmp_path / "hk_shift" / "Face1"
    base_dir.mkdir(parents=True)

    dicing_yaml = _write_dicing_metadata_yaml(
        tmp_path / "dicing_path_metadata.yaml",
        base_dir=base_dir,
        cal_file_path=cal_file,
        cutparams_filepath=cutparams_file,
        metrology_file_path=tmp_path / "dummy_metrology.dat",
        spindle="S1",
        cuttype="Thick",
        blade_diameter=100.0,
        orientation="Face1",
    )

    testtouch_yaml = _write_testtouch_yaml(
        tmp_path / "lens_testtouches.yaml",
        spindle="S1",
        orientation="Face1",
        x_center_shift="+0.250",
        x_postcal_shift="+0.125",
        y_postcal_shift="0",
        zcorr="-0.125",
    )

    # Build the no-shift HK directory using the actual wrapper
    housekeeping.generate_hkcut_files_config(
        p=[0.1, 0.2, 0.3],
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

    context = ch.get_cut_context(
        spindle="S1",
        orientation="Face1",
        dicing_metadata_path=str(dicing_yaml),
        testtouch_config_path=str(testtouch_yaml),
    )

    x_total_shift = context["x_total_shift"]
    zcorr = context["zcorr"]

    spindle_dir = Path(context["base_dir"]) / context["spindle"]
    noshift_dir = spindle_dir / "CutCammingThick-Noshift"
    shifted_dir = spindle_dir / "CutCammingThick"

    original_master = _ensure_2d(np.loadtxt(noshift_dir / "Master.txt"))
    assert original_master.shape[1] == 5

    # Check X pitch in the original Master against the cutparams file
    _, _, _, cutpitch = core_utils.get_cut_parameters(cutparams_file)
    original_xs = original_master[:, 1]
    if len(original_xs) > 1:
        assert np.allclose(np.diff(original_xs), cutpitch)

    # Apply the shift
    core_utils.shiftXZ_nocomp(
        directory=spindle_dir,
        ftype="Thick",
        xshift=x_total_shift,
        zshift=zcorr,
    )

    assert shifted_dir.exists()
    assert (shifted_dir / "Master.txt").exists()

    shifted_master = _ensure_2d(np.loadtxt(shifted_dir / "Master.txt"))
    assert shifted_master.shape == original_master.shape

    # Master columns are: num, x, ystart, z, yend
    assert np.allclose(shifted_master[:, 1], original_master[:, 1] + x_total_shift)
    assert np.allclose(shifted_master[:, 3], original_master[:, 3] + zcorr)

    # Pitch should be preserved after shift too
    shifted_xs = shifted_master[:, 1]
    if len(shifted_xs) > 1:
        assert np.allclose(np.diff(shifted_xs), cutpitch)

    original_cam_files = sorted(noshift_dir.glob("CutCamThick*.Cam"))
    shifted_cam_files = sorted(shifted_dir.glob("CutCamThick*.Cam"))

    assert len(original_cam_files) > 0
    assert len(original_cam_files) == len(shifted_cam_files)

    for orig_cam, shift_cam in zip(original_cam_files, shifted_cam_files):
        orig_pts = _ensure_2d(np.loadtxt(orig_cam, skiprows=4))
        shift_pts = _ensure_2d(np.loadtxt(shift_cam, skiprows=4))

        # CAM columns are: index, y, z
        assert orig_pts.shape == shift_pts.shape

        # Y should be unchanged
        assert np.allclose(shift_pts[:, 1], orig_pts[:, 1])

        # Z should be shifted by zcorr
        assert np.allclose(shift_pts[:, 2], orig_pts[:, 2] + zcorr)
