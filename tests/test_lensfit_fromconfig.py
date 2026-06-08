from unittest.mock import patch
import lensfit


def test_lensfit_fromconfig_calls_lensfit_with_expected_args():
    fake_dicing_metadata = {
        "orientations": {
            "0deg": {
                "lens_metrology_file_path": "/tmp/Convex/0deg/Lens_Met_0deg.dat"
            }
        }
    }

    fake_lensparams_cfg = {
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
        "alignment": {
            "x_rot_shift": 0.0
        },
        "cut": {
            "step_height": 7.046,
            "cut_diam": 459.0,
        }
    }

    with patch("lensfit.ch.load_yaml_config", side_effect=[fake_dicing_metadata, fake_lensparams_cfg]) as mock_load, \
         patch("lensfit.ch.get_lensparams_settings", return_value={"lensparams": fake_lensparams_cfg["lensparams"]}) as mock_lensparams, \
         patch("lensfit.lensfit") as mock_lensfit:

        lensfit.lensfit_fromconfig(
            orientation="0deg",
            dicing_metadata_path="config/dicing_path_metadata.yaml",
            lensparams_config_path="config/lensparams.yaml",
            afixed=0.1,
            bfixed=0.2,
            plot=False,
            return_full=True,
            verbose=False,
        )

        assert mock_load.call_count == 2
        mock_load.assert_any_call("config/dicing_path_metadata.yaml")
        mock_load.assert_any_call("config/lensparams.yaml")

        mock_lensparams.assert_called_once_with(fake_lensparams_cfg)

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
