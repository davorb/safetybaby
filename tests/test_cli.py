from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from babymon import app, audio, camera, config, factlog, notify, vision
from babymon.__main__ import main
from babymon.domain import (
    AlertKind,
    CameraStatus,
    DeliveryResult,
    Priority,
    SetControls,
    Still,
    Tick,
    VisionReason,
)
from tests.conftest import at, observation, seen

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = object()
STOMACH = observation(body="stomach")

NIGHT = [
    CameraStatus(at(0), up=True, detail="up"),
    Tick(at(1)),
    Tick(at(2)),
    seen(5, STOMACH),
    Tick(at(6)),
    seen(20, STOMACH, VisionReason.CONFIRM),
    Tick(at(21)),
    DeliveryResult(at(22), AlertKind.ON_STOMACH, delivered=True),
    Tick(at(22)),
    Tick(at(23)),
]


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["run"],
        ["probe"],
        ["probe", "camera"],
        ["probe", "audio"],
        ["probe", "vision"],
        ["probe", "alert"],
        ["reference"],
        ["replay"],
    ],
    ids=" ".join,
)
def test_help_for_every_subcommand(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([*argv, "--help"])
    assert exit_info.value.code == 0
    assert "usage: babymon" in capsys.readouterr().out


def test_help_and_replay_import_none_of_the_actor_modules(tmp_path):
    facts = tmp_path / "facts.jsonl"
    facts.write_text("\n".join(map(factlog.encode, NIGHT)) + "\n")
    code = (
        "import sys; from babymon.__main__ import main\n"
        f"main(['replay', {str(facts)!r}])\n"
        "loaded = {'babymon.camera', 'babymon.audio', 'babymon.vision', 'babymon.notify',\n"
        "          'babymon.app', 'babymon.store', 'anthropic', 'cv2', 'onnxruntime'}\n"
        "assert not loaded & set(sys.modules), loaded & set(sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_replay_prints_the_effects_of_a_ten_line_night(tmp_path, capsys):
    facts = tmp_path / "facts.jsonl"
    facts.write_text("\n".join(map(factlog.encode, NIGHT)) + "\n")
    assert main(["replay", str(facts)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [line.split("  ", 1)[1].split(" body=")[0] for line in lines] == [
        "record camera up=True up",
        "request_vision periodic",
        "record vision periodic",
        "record vision confirm",
        "notify emergency on_stomach 'Baby on stomach'",
    ]
    assert [line.split("  ")[0][11:] for line in lines] == [
        "02:00:00Z",
        "02:00:01Z",
        "02:00:05Z",
        "02:00:20Z",
        "02:00:21Z",
    ]


def test_replay_names_the_bad_line(tmp_path):
    facts = tmp_path / "facts.jsonl"
    facts.write_text(factlog.encode(Tick(at(1))) + "\n{oops\n")
    with pytest.raises(SystemExit, match=r"facts\.jsonl:2"):
        main(["replay", str(facts)])


def test_replay_of_a_missing_file_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["replay", "/no/such.jsonl"])
    assert exit_info.value.code == 2
    assert "no such file" in capsys.readouterr().err


@pytest.fixture
def owners(monkeypatch):
    """Replace every function the CLI hands work to; record how it was called."""
    calls: list[tuple] = []

    def fake(name):
        async def call(*args):
            calls.append((name, *args))

        return call

    monkeypatch.setattr(config, "load", lambda: SETTINGS)
    for module, name in [
        (app, "run"),
        (camera, "probe"),
        (camera, "grab_still"),
        (audio, "probe"),
        (vision, "probe"),
        (notify, "probe"),
    ]:
        monkeypatch.setattr(
            module, name, fake(f"{module.__name__.split('.')[-1]}.{name}"), raising=False
        )
    return calls


@pytest.mark.parametrize(
    ("flags", "armed", "nap_mode"),
    [
        (["--nap"], None, True),
        (["--no-nap"], None, False),
        (["--disarm"], False, None),
        (["--arm", "--nap"], True, True),
    ],
)
def test_run_flags_become_one_set_controls(owners, flags, armed, nap_mode):
    before = datetime.now(UTC)
    assert main(["run", *flags]) == 0
    [(name, settings, controls, record)] = owners
    assert (name, settings, record) == ("app.run", SETTINGS, None)
    assert (controls.armed, controls.nap_mode) == (armed, nap_mode)
    assert isinstance(controls, SetControls) and controls.at >= before


def test_run_without_flags_leaves_persisted_controls_alone(owners, tmp_path):
    main(["run", "--record-facts", str(tmp_path / "night.jsonl")])
    assert owners == [("app.run", SETTINGS, None, tmp_path / "night.jsonl")]


@pytest.mark.parametrize("flags", [["--nap", "--no-nap"], ["--arm", "--disarm"]])
def test_contradicting_flags_are_rejected(owners, flags):
    with pytest.raises(SystemExit) as exit_info:
        main(["run", *flags])
    assert exit_info.value.code == 2
    assert owners == []


def test_probes_pass_their_arguments_to_the_owners(owners, tmp_path):
    wav, jpg, ref = tmp_path / "a.wav", tmp_path / "f.jpg", tmp_path / "r.jpg"
    for p in (wav, jpg, ref):
        p.write_bytes(b"x")
    main(["probe", "camera", "--seconds", "5"])
    main(["probe", "camera"])
    main(["probe", "audio", str(wav)])
    main(["probe", "vision", str(jpg), "--reference", str(ref)])
    main(["probe", "vision", str(jpg)])
    main(["probe", "alert", "--priority", "emergency"])
    assert owners == [
        ("camera.probe", SETTINGS, 5.0),
        ("camera.probe", SETTINGS, 30.0),
        ("audio.probe", SETTINGS, wav),
        ("vision.probe", SETTINGS, jpg, ref),
        ("vision.probe", SETTINGS, jpg, None),
        ("notify.probe", SETTINGS, Priority.EMERGENCY),
    ]


def test_probe_rejects_a_missing_file_and_an_unknown_priority(owners, capsys):
    for argv in (["probe", "audio", "/no.wav"], ["probe", "alert", "--priority", "2"]):
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 2
    assert owners == []


def test_reference_stores_a_grabbed_still(monkeypatch, tmp_path, capsys):
    from babymon import store

    still = Still(b"jpeg", at(7))
    stored: list[Still] = []

    class FakeStore:
        @classmethod
        def open(cls, data_dir):
            assert data_dir == tmp_path
            return cls()

        def set_reference(self, s):
            stored.append(s)

    settings = SimpleNamespace(data_dir=tmp_path)

    async def grab(given):
        assert given is settings
        return still

    monkeypatch.setattr(config, "load", lambda: settings)
    monkeypatch.setattr(camera, "grab_still", grab, raising=False)
    monkeypatch.setattr(store, "Store", FakeStore)
    assert main(["reference"]) == 0
    assert stored == [still]
    assert "reference stored" in capsys.readouterr().out
