"""Settings parse from .env in the working directory and from the environment."""

from __future__ import annotations

import os

import pytest

from babymon.config import load
from babymon.domain import Capabilities

CAMERA = "BABYMON_CAMERA__USER=cam\nBABYMON_CAMERA__PASSWORD=p@ss:w\n"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("BABYMON_") or name == "ANTHROPIC_API_KEY":
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


def write_env(text: str) -> None:
    with open(".env", "w") as f:
        f.write(text)


def test_full_env_parses():
    write_env(
        CAMERA + "BABYMON_CAMERA__MOTION_ROI=10,20,300,200\n"
        "BABYMON_VISION__API_KEY=sk-1\n"
        "BABYMON_PUSHOVER__APP_TOKEN=tok\n"
        "BABYMON_PUSHOVER__USER_KEYS=u1, u2\n"
        "BABYMON_TUNING__HEAD_LEFT_ALERT_S=90\n"
        "BABYMON_HTTP_PORT=9000\n"
    )
    s = load()
    assert s.camera.rtsp_url() == "rtsp://cam:p%40ss%3Aw@192.168.68.75:554/stream2"
    assert s.camera.motion_roi == (10, 20, 300, 200)
    assert s.vision is not None and s.vision.api_key.get_secret_value() == "sk-1"
    assert s.pushover is not None and s.pushover.user_keys == ("u1", "u2")
    assert s.tuning.head_left_alert_s == 90.0
    assert s.http_port == 9000
    assert s.capabilities() == Capabilities(vision=True, push=True)


def test_absent_optional_groups_disable_capabilities():
    write_env(CAMERA)
    assert load().capabilities() == Capabilities(vision=False, push=False)


def test_partial_pushover_group_names_the_missing_variable(capsys):
    write_env(CAMERA + "BABYMON_PUSHOVER__APP_TOKEN=tok\n")
    with pytest.raises(SystemExit) as exit_info:
        load()
    assert exit_info.value.code == 2
    assert "BABYMON_PUSHOVER__USER_KEYS" in capsys.readouterr().err


def test_missing_camera_names_each_required_variable(capsys):
    with pytest.raises(SystemExit):
        load()
    err = capsys.readouterr().err
    assert "BABYMON_CAMERA__USER" in err and "BABYMON_CAMERA__PASSWORD" in err


def test_invalid_value_names_the_variable(capsys):
    write_env(CAMERA + "BABYMON_TUNING__UNSAFE_STREAK=two\n")
    with pytest.raises(SystemExit):
        load()
    assert "BABYMON_TUNING__UNSAFE_STREAK" in capsys.readouterr().err


def test_anthropic_api_key_alias_from_environment(monkeypatch):
    write_env(CAMERA)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    s = load()
    assert s.vision is not None and s.vision.api_key.get_secret_value() == "sk-env"


def test_anthropic_api_key_alias_from_dotenv():
    write_env(CAMERA + "ANTHROPIC_API_KEY=sk-file\n")
    s = load()
    assert s.vision is not None and s.vision.api_key.get_secret_value() == "sk-file"


def test_explicit_vision_key_beats_the_alias(monkeypatch):
    write_env(CAMERA + "BABYMON_VISION__API_KEY=sk-explicit\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    s = load()
    assert s.vision is not None and s.vision.api_key.get_secret_value() == "sk-explicit"


def test_environment_beats_dotenv(monkeypatch):
    write_env(CAMERA + "BABYMON_HTTP_PORT=9000\n")
    monkeypatch.setenv("BABYMON_HTTP_PORT", "9001")
    assert load().http_port == 9001
