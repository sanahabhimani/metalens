import pytest
from unittest.mock import patch

import lensfit


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_context():
    """Minimal context dict returned by ch.get_cut_context."""
    return {
        "base_dir": "/data/lens/face1",
        "blade_diameter": 100.0,
        "cal_file_path": "/cal/spindle_cal.txt",
        "cutparams_filepath": "cutparams.txt",
        "type": "Thick",
        "lensparams": {
            "R": -818.9215895,
            "k": -30.00046667,
            "a1": -1.56460553e-4,
            "a2": -3.06349779e-9,
            "a3": 5.00246955e-14,
            "a4": 0.0,
            "t_ctr": 53.507,
            "diam": 448.0,
        },
        "cut_diam": 459,
        "step_height": 7.046,
        "x_rot_shift": 0.0,
    }


def _call_config(
    p=None,
    p2=None,
    spindle="S1",
    orientation="Face1",
    dicing_metadata_path="/cfg/dicing_path_metadata.yaml",
    lensparams_config_path="/cfg/lensparams.yaml",
    afixed=0.1,
    bfixed=0.2,
    use_fit="p",
    yres=0.500,
):
    """Helper to invoke generate_lens_cutfiles_fromconfig with sensible defaults."""
    if p is None:
        p = [0.1, 0.2, 0.3, 0.4, 0.5]
    if p2 is None:
        p2 = [1.1, 1.2, 1.3, 1.4, 1.5]

    return lensfit.generate_lens_cutfiles_fromconfig(
        p=p,
        p2=p2,
        spindle=spindle,
        orientation=orientation,
        dicing_metadata_path=dicing_metadata_path,
        lensparams_config_path=lensparams_config_path,
        afixed=afixed,
        bfixed=bfixed,
        use_fit=use_fit,
        yres=yres,
    )


# ---------------------------------------------------------------------------
# Test 1: generate_lens_cut_files is called with all parameters from config_helper
# ---------------------------------------------------------------------------

def test_calls_generate_lens_cut_files_with_context_params(mock_context):
    """generate_lens_cutfiles_fromconfig correctly calls generate_lens_cut_files
    using values resolved by ch.get_cut_context."""
    p = [0.1, 0.2, 0.3, 0.4, 0.5]
    p2 = [1.1, 1.2, 1.3, 1.4, 1.5]

    with patch("lensfit.ch.get_cut_context", return_value=mock_context) as mock_ctx, \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config(
            p=p,
            p2=p2,
            spindle="S1",
            orientation="Face1",
            dicing_metadata_path="/cfg/dicing_path_metadata.yaml",
            lensparams_config_path="/cfg/lensparams.yaml",
            afixed=0.1,
            bfixed=0.2,
            use_fit="p",
            yres=0.500,
        )

        mock_ctx.assert_called_once_with(
            spindle="S1",
            orientation="Face1",
            dicing_metadata_path="/cfg/dicing_path_metadata.yaml",
            lensparams_config_path="/cfg/lensparams.yaml",
        )

        mock_gen.assert_called_once_with(
            p=p,
            p2=p2,
            pathname="/data/lens/face1/",
            spindle="S1",
            calibrationfilepath="/cal/spindle_cal.txt",
            cutparamsfile="cutparams.txt",
            cutdiameter=459.0,
            bladeradius=50.0,
            cuttype="Thick",
            lensparams=[
                -818.9215895,
                -30.00046667,
                -1.56460553e-4,
                -3.06349779e-9,
                5.00246955e-14,
                0.0,
                53.507,
                448.0,
            ],
            afixed=0.1,
            bfixed=0.2,
            stepheight=7.046,
            use_fit="p",
            x_rot_shift=0.0,
            yres=0.500,
        )


# ---------------------------------------------------------------------------
# Test 2: base_dir trailing-slash normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_dir,expected_pathname", [
    ("/data/lens/face1",  "/data/lens/face1/"),
    ("/data/lens/face1/", "/data/lens/face1/"),
])
def test_handles_base_dir_trailing_slash(mock_context, base_dir, expected_pathname):
    """pathname passed to generate_lens_cut_files always ends with exactly one '/'."""
    mock_context["base_dir"] = base_dir

    with patch("lensfit.ch.get_cut_context", return_value=mock_context), \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config()

        assert mock_gen.call_args.kwargs["pathname"] == expected_pathname


# ---------------------------------------------------------------------------
# Test 3: bladeradius is blade_diameter / 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blade_diameter,expected_radius", [
    (100.0, 50.0),
    (54.0,  27.0),
    (1.0,   0.5),
    (0.8,   0.4),
    ("100.0", 50.0),
    ("54.0",  27.0),
])
def test_calculates_bladeradius_from_blade_diameter(mock_context, blade_diameter, expected_radius):
    """bladeradius passed to generate_lens_cut_files equals blade_diameter / 2."""
    mock_context["blade_diameter"] = blade_diameter

    with patch("lensfit.ch.get_cut_context", return_value=mock_context), \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config()

        assert mock_gen.call_args.kwargs["bladeradius"] == expected_radius


# ---------------------------------------------------------------------------
# Test 4: lensparams are flattened in SawPy order
# ---------------------------------------------------------------------------

def test_flattens_lensparams_dict_into_expected_order(mock_context):
    """lensparams dict from config is converted into the ordered list
    expected by generate_lens_cut_files."""
    mock_context["lensparams"] = {
        "R": 10.0,
        "k": 20.0,
        "a1": 30.0,
        "a2": 40.0,
        "a3": 50.0,
        "a4": 60.0,
        "t_ctr": 70.0,
        "diam": 80.0,
    }

    with patch("lensfit.ch.get_cut_context", return_value=mock_context), \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config()

        assert mock_gen.call_args.kwargs["lensparams"] == [
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0
        ]


# ---------------------------------------------------------------------------
# Test 5: afixed, bfixed, use_fit, and yres forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("afixed,bfixed,use_fit,yres", [
    (0.1,   0.2,   "p",  0.500),
    (1.25, -0.75,  "p2", 0.250),
    (0.0,   0.0,   "p",  1.000),
])
def test_passes_fit_and_resolution_params(mock_context, afixed, bfixed, use_fit, yres):
    """afixed, bfixed, use_fit, and yres are forwarded unchanged
    to generate_lens_cut_files."""
    with patch("lensfit.ch.get_cut_context", return_value=mock_context), \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config(
            afixed=afixed,
            bfixed=bfixed,
            use_fit=use_fit,
            yres=yres,
        )

        kwargs = mock_gen.call_args.kwargs
        assert kwargs["afixed"] == afixed
        assert kwargs["bfixed"] == bfixed
        assert kwargs["use_fit"] == use_fit
        assert kwargs["yres"] == yres


# ---------------------------------------------------------------------------
# Test 6: cutdiameter, stepheight, and x_rot_shift convert cleanly to float
# ---------------------------------------------------------------------------

def test_converts_numeric_context_values_to_float(mock_context):
    """cutdiameter, stepheight, and x_rot_shift are passed as floats
    even if config_helper returns numeric strings."""
    mock_context["cut_diam"] = "459"
    mock_context["step_height"] = "7.046"
    mock_context["x_rot_shift"] = "0.125"

    with patch("lensfit.ch.get_cut_context", return_value=mock_context), \
         patch("lensfit.generate_lens_cut_files") as mock_gen:

        _call_config()

        kwargs = mock_gen.call_args.kwargs
        assert kwargs["cutdiameter"] == 459.0
        assert kwargs["stepheight"] == 7.046
        assert kwargs["x_rot_shift"] == 0.125
