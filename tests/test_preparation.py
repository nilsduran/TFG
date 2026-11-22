from br_analysis.preparation import preparation_specs


def test_all_models_and_temporal_onset_share_four_preprocessing_variants() -> None:
    specs = preparation_specs()

    assert len(specs) == 4
    assert {model for spec in specs for model in spec.model_ids} == {
        "lr_theta",
        "lr_alpha",
        "riemann",
        "svc_agnostic",
        "svc_clustering",
        "eegnet_agnostic",
        "eegnet_clustering",
        "st_gnn",
        "st_eegformer",
        "labram",
        "temporal_onset",
    }
    onset = next(spec for spec in specs if spec.preset_id == "erp")
    assert onset.analysis_window == (-0.74, 1.5)
    assert onset.resample_sfreq == 250.0


def test_preparation_can_be_limited_to_models() -> None:
    specs = preparation_specs(["lr_theta", "lr_alpha"])

    assert len(specs) == 1
    assert specs[0].model_ids == ("lr_theta", "lr_alpha")
    assert specs[0].analysis_window == (0.0, 1.5)
