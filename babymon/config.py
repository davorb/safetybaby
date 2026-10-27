"""Settings: the one boundary for environment variables and .env.

Parsed once at startup into frozen, typed groups. Optional groups (vision, pushover)
are all-or-nothing: a partly set group is a startup error, a fully absent one disables
that capability and the dashboard says so.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from babymon.domain import Capabilities, Tuning


class CameraConfig(BaseModel, frozen=True):
    host: str = "192.168.68.75"
    port: int = 554
    user: str
    password: SecretStr
    stream_path: str = "/stream2"
    stall_s: float = 10.0
    motion_roi: tuple[int, int, int, int] | None = None
    """x, y, w, h in 640x360 frame coordinates. None = whole frame."""
    motion_noise_threshold: int = 12

    def rtsp_url(self) -> str:
        raise NotImplementedError


class VisionConfig(BaseModel, frozen=True):
    api_key: SecretStr
    model: str = "claude-opus-5-5"


class PushoverConfig(BaseModel, frozen=True):
    app_token: SecretStr
    user_keys: tuple[str, ...]
    emergency_retry_s: int = 30
    emergency_expire_s: int = 600


class Settings(BaseSettings):
    """Env names: BABYMON_CAMERA__USER, BABYMON_VISION__API_KEY, BABYMON_TUNING__HEAD_LEFT_ALERT_S, ...
    ANTHROPIC_API_KEY is also accepted for vision.api_key."""

    model_config = SettingsConfigDict(
        env_prefix="BABYMON_", env_nested_delimiter="__", env_file=".env", frozen=True
    )

    camera: CameraConfig
    vision: VisionConfig | None = None
    pushover: PushoverConfig | None = None
    tuning: Tuning = Tuning()
    data_dir: Path = Path("/data")
    yamnet_path: Path = Path("/models/yamnet.onnx")
    http_port: int = 8080

    def capabilities(self) -> Capabilities:
        return Capabilities(vision=self.vision is not None, push=self.pushover is not None)


def load() -> Settings:
    """Parse or exit with a message naming each missing or invalid variable."""
    raise NotImplementedError
