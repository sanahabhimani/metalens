import lensfit
from unittest.mock import patch


def test_lensfit_fromconfig_calls_lensfit_with_expected_args():
    fake_context = {
        "metrology_file_path": "/tmp/Convex/0deg/Lens_Met_0deg.dat",
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
    }

    with patch("lensfit.ch.get_cut_context", return_value=fake_context) as mock_ctx, \
         patch("lensfit.lensfit") as mock_lensfit:

        lensfit.lensfit_fromconfig(
            spindle="SpindleB",
            orientation="0deg",
            dicing_metadata_path="config/dicing_path_metadata.yaml",
            lensparams_config_path="config/lensparams.yaml",
            afixed=0.1,
            bfixed=0.2,
            plot=False,
            return_full=True,
            verbose=False,
        )

        mock_ctx.assert_called_once_with(
            spindle="SpindleB",
            orientation="0deg",
            dicing_metadata_path="config/dicing_path_metadata.yaml",
            testtouch_config_path=None,
            lensparams_config_path="config/lensparams.yaml",
        )

        mock_lensfit.assert_called_once_with(
            pathname="/tmp/Convex/0deg/",
            metrologyfilename="Lens_Met_0deg.dat",
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
            plot=False,
            return_full=True,
            verbose=False,
        )
