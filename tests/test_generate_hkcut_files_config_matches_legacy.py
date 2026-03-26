import pytest
from unittest.mock import patch, call

import housekeeping


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
    }


def _call_config(p=None, spindle="S1", orientation="Face1",
                 dicing_metadata_path="/cfg/spindles.yaml",
                 testtouch_config_path="/cfg/testtouch.yaml",
                 xstart=-50.0, xend=50.0,
                 ystart=-30.0, yend=30.0,
                 use_noshift_suffix=True):
    """Helper to invoke generate_hkcut_files_config with sensible defaults."""
    if p is None:
        p = [0.1, 0.2, 0.3]
    return housekeeping.generate_hkcut_files_config(
        p=p,
        spindle=spindle,
        orientation=orientation,
        dicing_metadata_path=dicing_metadata_path,
        testtouch_config_path=testtouch_config_path,
        xstart=xstart,
        xend=xend,
        ystart=ystart,
        yend=yend,
        use_noshift_suffix=use_noshift_suffix,
    )


# ---------------------------------------------------------------------------
# Test 1: generate_hkcut_files is called with all parameters from config_helper
# ---------------------------------------------------------------------------

def test_calls_generate_hkcut_files_with_context_params(mock_context):
    """generate_hkcut_files_config correctly calls generate_hkcut_files
    using values resolved by ch.get_cut_context."""
    p = [0.1, 0.2, 0.3]

    with patch("housekeeping.ch.get_cut_context", return_value=mock_context) as mock_ctx, \
         patch("housekeeping.generate_hkcut_files") as mock_gen:

        _call_config(
            p=p,
            spindle="S1",
            orientation="Face1",
            dicing_metadata_path="/cfg/spindles.yaml",
            testtouch_config_path="/cfg/testtouch.yaml",
            xstart=-50.0,
            xend=50.0,
            ystart=-30.0,
            yend=30.0,
        )

        # config_helper was asked for the right spindle / orientation / paths
        mock_ctx.assert_called_once_with(
            spindle="S1",
            orientation="Face1",
            dicing_metadata_path="/cfg/spindles.yaml",
            testtouch_config_path="/cfg/testtouch.yaml",
        )

        # generate_hkcut_files received every resolved value
        mock_gen.assert_called_once_with(
            p=p,
            pathname="/data/lens/face1/",
            spindle="S1",
            calibrationfilepath="/cal/spindle_cal.txt",
            cutparamsfile="cutparams.txt",
            bladeradius=50.0,
            cuttype="Thick",
            xstart=-50.0,
            xend=50.0,
            ystart=-30.0,
            yend=30.0,
            use_noshift_suffix=True,
        )


# ---------------------------------------------------------------------------
# Test 2: base_dir trailing-slash normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("base_dir,expected_pathname", [
    ("/data/lens/face1",  "/data/lens/face1/"),   # no trailing slash → one added
    ("/data/lens/face1/", "/data/lens/face1/"),   # trailing slash present → unchanged
])
def test_handles_base_dir_trailing_slash(mock_context, base_dir, expected_pathname):
    """pathname passed to generate_hkcut_files always ends with exactly one '/'."""
    mock_context["base_dir"] = base_dir

    with patch("housekeeping.ch.get_cut_context", return_value=mock_context), \
         patch("housekeeping.generate_hkcut_files") as mock_gen:

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
])
def test_calculates_bladeradius_from_blade_diameter(mock_context, blade_diameter, expected_radius):
    """bladeradius passed to generate_hkcut_files equals blade_diameter / 2."""
    mock_context["blade_diameter"] = blade_diameter

    with patch("housekeeping.ch.get_cut_context", return_value=mock_context), \
         patch("housekeeping.generate_hkcut_files") as mock_gen:

        _call_config()

        assert mock_gen.call_args.kwargs["bladeradius"] == expected_radius


# ---------------------------------------------------------------------------
# Test 4: x, y, and use_noshift_suffix parameters forwarded correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("xstart,xend,ystart,yend,use_noshift_suffix", [
    (-50.0,  50.0, -30.0,  30.0, True),
    (  0.0, 100.0, -10.0,  10.0, False),
    (-75.5,  75.5, -75.5,  75.5, True),
])
def test_passes_x_y_and_noshift_suffix_params(
    mock_context, xstart, xend, ystart, yend, use_noshift_suffix
):
    """xstart, xend, ystart, yend, and use_noshift_suffix are forwarded
    unchanged to generate_hkcut_files."""
    with patch("housekeeping.ch.get_cut_context", return_value=mock_context), \
         patch("housekeeping.generate_hkcut_files") as mock_gen:

        _call_config(
            xstart=xstart,
            xend=xend,
            ystart=ystart,
            yend=yend,
            use_noshift_suffix=use_noshift_suffix,
        )

        kwargs = mock_gen.call_args.kwargs
        assert kwargs["xstart"] == xstart
        assert kwargs["xend"] == xend
        assert kwargs["ystart"] == ystart
        assert kwargs["yend"] == yend
        assert kwargs["use_noshift_suffix"] == use_noshift_suffix

@pytest.mark.parametrize("blade_diameter,expected_radius", [
    ("100.0", 50.0),
    ("54.0", 27.0),
])
def test_hk_config_converts_string_blade_diameter(mock_context, blade_diameter, expected_radius):
    mock_context["blade_diameter"] = blade_diameter

    with patch("housekeeping.ch.get_cut_context", return_value=mock_context), \
         patch("housekeeping.generate_hkcut_files") as mock_gen:

        _call_config()

        assert mock_gen.call_args.kwargs["bladeradius"] == expected_radius
