import numpy as np
from pathlib import Path

import lensfit


# ---------------------------------------------------------------------------
# Small file writers
# ---------------------------------------------------------------------------

def _write_dicing_metadata_yaml(
    path,
    base_dir,
    cal_file_path,
    cutparams_filepath,
    metrology_file_path,
    spindle="S1",
    cuttype="Thick",
    blade_diameter=100.0,
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
        f"    lens_metrology_file_path: '{metrology_file_path}'\n"
        f"    base_dir: '{base_dir}'\n"
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


def _write_metrology_file(path):
    """
    Write a tiny synthetic metrology .dat file with columns:
    x, y, z, r
    """
    path = Path(path)

    pts = np.array([
        [82.0, 364.0, -39.5, 0.5],
        [83.0, 365.0, -39.8, 0.5],
        [84.0, 366.0, -40.1, 0.5],
        [82.5, 365.5, -39.7, 0.5],
        [83.5, 364.5, -39.9, 0.5],
        [84.5, 365.5, -40.2, 0.5],
    ], dtype=float)

    np.savetxt(path, pts, delimiter=",")
    return path


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_lensfit_fromconfig_matches_legacy(tmp_path):
    """
    Confirm that lensfit_fromconfig(...) produces the same fit outputs as a
    direct legacy call to lensfit(...), using the same metrology file and
    lens parameters.
    """
    inputs_dir = tmp_path / "inputs"
    inputs_dir.mkdir()

    # These two are not used by lensfit itself, but the dicing metadata file
    # currently expects them to exist as part of the shared paths block.
    cal_file = inputs_dir / "SpindleCal.txt"
    cal_file.write_text("Spindle Xoffset Yoffset Zoffset\nS1 1.0 2.0 3.0\nS2 4.0 5.0 6.0\n")

    cutparams_file = inputs_dir / "cutparams.txt"
    cutparams_file.write_text("value\n1.0\n0.5\n0.25\n4.0\n")

    metrology_file = _write_metrology_file(inputs_dir / "Lens_Met_0deg.dat")

    dicing_yaml = _write_dicing_metadata_yaml(
        tmp_path / "dicing_path_metadata.yaml",
        base_dir=tmp_path / "dummy_base_dir",
        cal_file_path=cal_file,
        cutparams_filepath=cutparams_file,
        metrology_file_path=metrology_file,
        spindle="S1",
        cuttype="Thick",
        blade_diameter=100.0,
        orientation="0deg",
    )

    lensparams_yaml = _write_lensparams_yaml(
        tmp_path / "lensparams.yaml"
    )

    afixed = 0.1
    bfixed = 0.2

    lensparams = [
        -818.9215895,
        -30.00046667,
        -1.56460553e-4,
        -3.06349779e-9,
        5.00246955e-14,
        0.0,
        53.507,
        448.0,
    ]

    # -----------------------------------------------------------------------
    # Legacy call
    # -----------------------------------------------------------------------
    p_legacy, p2_legacy = lensfit.lensfit(
        pathname=str(metrology_file.parent) + "/",
        metrologyfilename=metrology_file.name,
        lensparams=lensparams,
        afixed=afixed,
        bfixed=bfixed,
        plot=False,
        return_full=False,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Config-driven call
    # -----------------------------------------------------------------------
    p_cfg, p2_cfg = lensfit.lensfit_fromconfig(
        orientation="0deg",
        dicing_metadata_path=str(dicing_yaml),
        lensparams_config_path=str(lensparams_yaml),
        afixed=afixed,
        bfixed=bfixed,
        plot=False,
        return_full=False,
        verbose=False,
    )

    # -----------------------------------------------------------------------
    # Compare fit outputs
    # -----------------------------------------------------------------------
    assert np.allclose(p_legacy, p_cfg)
    assert np.allclose(p2_legacy, p2_cfg)
