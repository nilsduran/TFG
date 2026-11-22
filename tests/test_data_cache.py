import json
from dataclasses import replace

import numpy as np

import br_analysis.data as data_module
from br_analysis.config import ProjectPaths
from br_analysis.dataset import SubjectExclusion
from br_analysis.preprocessing import get_preprocessing_preset
from br_analysis.tasks import get_task


def _preprocessed_epochs() -> data_module._PreprocessedEpochs:
    events = np.array([1, 3, 2, 5, 1, 4, 2, 6], dtype=np.int16)
    return data_module._PreprocessedEpochs(
        X=np.arange(8 * 2 * 4, dtype=np.float32).reshape(8, 2, 4),
        event_codes=events,
        subjects=np.repeat(["P01", "P02"], 4),
        trial_indices=np.tile(np.arange(4), 2),
        channel_names=["Cz", "Oz"],
        times=np.arange(4, dtype=np.float32),
        sfreq=4.0,
        exclusions=[],
    )


def test_tasks_and_channel_picks_are_derived_from_shared_preprocessing(
    tmp_path,
    monkeypatch,
) -> None:
    shared = _preprocessed_epochs()
    monkeypatch.setattr(
        data_module,
        "_iter_preprocessed_subjects",
        lambda *args, **kwargs: iter([(shared, None)]),
    )
    task = replace(get_task("c_vs_ip"), min_trials_per_class=1)
    paths = ProjectPaths.from_env("test", project_dir=tmp_path, create_dirs=False)

    dataset = data_module.load_epoch_dataset(
        paths,
        task,
        "broadband",
        picks=["Oz"],
        verbose=False,
    )

    assert dataset.channel_names == ["Oz"]
    assert dataset.X.shape == (8, 1, 4)
    assert dataset.class_counts == {"c": 4, "ip": 4}


def test_preprocessing_cache_round_trip(tmp_path) -> None:
    expected = _preprocessed_epochs()
    expected.exclusions.append(SubjectExclusion("P03", "read error", "broken"))
    data_path = tmp_path / "epochs.npz"
    metadata_path = tmp_path / "epochs.json"

    data_module._save_cached_preprocessing(expected, data_path, metadata_path)
    actual = data_module._load_cached_preprocessing(data_path, metadata_path)

    np.testing.assert_array_equal(actual.X, expected.X)
    np.testing.assert_array_equal(actual.event_codes, expected.event_codes)
    assert actual.channel_names == expected.channel_names
    assert actual.exclusions == expected.exclusions


def test_preprocessing_cache_name_is_readable_and_fingerprinted(tmp_path) -> None:
    fif_path = tmp_path / "P002.fif"
    fif_path.write_bytes(b"test")
    paths = ProjectPaths.from_env("test", project_dir=tmp_path, create_dirs=False)

    data_path, metadata_path = data_module._preprocessing_cache_paths(
        paths,
        fif_path,
        get_preprocessing_preset("band_limited"),
        (0.0, 1.5),
    )

    assert data_path.name.startswith("P002__band_limited__0p0-1p5__")
    assert len(data_path.stem.rsplit("__", 1)[1]) == 12
    assert metadata_path.with_suffix(".npz") == data_path

    broadband_path, _ = data_module._preprocessing_cache_paths(
        paths,
        fif_path,
        get_preprocessing_preset("broadband"),
        (0.0, 1.5),
    )
    pretrained_path, _ = data_module._preprocessing_cache_paths(
        paths,
        fif_path,
        get_preprocessing_preset("pretrained"),
        (0.0, 1.5),
    )
    assert pretrained_path == broadband_path


def test_portable_preprocessing_loads_without_fif_files(tmp_path) -> None:
    data_dir = tmp_path / "portable"
    data_dir.mkdir()
    cached = _preprocessed_epochs()
    cached.times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
    data_path = data_dir / "P01.npz"
    metadata_path = data_dir / "P01.json"
    data_module._save_cached_preprocessing(
        cached,
        data_path,
        metadata_path,
    )
    (data_dir / "preprocessed-manifest.json").write_text(
        json.dumps(
            {
                "format_version": data_module.PORTABLE_PREPROCESSING_VERSION,
                "variants": [
                    {
                        "id": "broadband",
                        "analysis_window": [0.0, 1.5],
                        "preprocessing": {
                            "bandpass": [1.0, 40.0],
                            "resample_sfreq": 128.0,
                        },
                        "subjects": [
                            {
                                "subject": "P01",
                                "data_file": data_path.name,
                                "metadata_file": metadata_path.name,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    paths = ProjectPaths.from_env(
        "test",
        project_dir=tmp_path,
        data_dir=data_dir,
        create_dirs=False,
    )
    task = replace(get_task("c_vs_ip"), min_trials_per_class=1)

    dataset = data_module.load_epoch_dataset(
        paths,
        task,
        "pretrained",
        verbose=False,
    )

    assert dataset.X.shape == (8, 2, 4)
    assert dataset.class_counts == {"c": 4, "ip": 4}
