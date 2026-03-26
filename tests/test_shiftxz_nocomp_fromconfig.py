from pathlib import Path
from unittest.mock import patch
import core_utils


def test_shiftxz_nocomp_fromconfig_calls_shift_function_with_expected_values():
    fake_context = {
        "base_dir": "/tmp/Convex/0deg",
        "spindle": "SpindleB",
        "type": "Thick",
        "x_total_shift": 0.010,
        "zcorr": -0.02834,
    }

    with patch("core_utils.ch.get_cut_context", return_value=fake_context) as mock_ctx, \
         patch("core_utils.shiftXZ_nocomp") as mock_shift:

        core_utils.shiftXZ_nocomp_fromconfig(
            spindle="SpindleB",
            orientation="0deg",
            dicing_metadata_path="config/dicing_path_metadata.yaml",
            testtouch_config_path="config/lens_testtouches.yaml",
        )

        mock_ctx.assert_called_once_with(
            spindle="SpindleB",
            orientation="0deg",
            dicing_metadata_path="config/dicing_path_metadata.yaml",
            testtouch_config_path="config/lens_testtouches.yaml",
        )

        mock_shift.assert_called_once_with(
            directory=Path("/tmp/Convex/0deg") / "SpindleB",
            ftype="Thick",
            xshift=0.010,
            zshift=-0.02834,
        )
