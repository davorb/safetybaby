"""Settings: the one boundary for environment variables and .env.

Parsed once at startup into frozen, typed groups. Optional groups (vision, pushover)
are all-or-nothing: a partly set group is a startup error, a fully absent one disables
that capability and the dashboard says so.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr, ValidationError, field_validator
from pydantic_settings import (
    BaseSettings,
    NoDecode,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from babymon.domain import Capabilities, Tuning

ENV_FILE = ".env"
ENV_PREFIX = "BABYMON_"
VISION_KEY_ALIAS = "ANTHROPIC_API_KEY"


def _split_csv(value: Any) -> Any:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


class CameraConfig(BaseModel, frozen=True):
    host: str = "192.168.68.75"
    port: int = 554
    user: str
    password: SecretStr
    stream_path: str = "/stream2"
    stall_s: float = 10.0
    motion_roi: Annotated[tuple[int, int, int, int] | None, NoDecode] = None
    """x, y, w, h in 640x360 frame coordinates. None = whole frame."""
    motion_noise_threshold: int = 12

    @field_validator("motion_roi", mode="before")
    @classmethod
    def _roi_from_csv(cls, value: Any) -> Any:
        return _split_csv(value)

    def rtsp_url(self) -> str:
        user = quote(self.user, safe="")
        password = quote(self.password.get_secret_value(), safe="")
        return f"rtsp://{user}:{password}@{self.host}:{self.port}{self.stream_path}"


class VisionConfig(BaseModel, frozen=True):
    api_key: SecretStr
    model: str = "claude-haiku-5-5"


class PushoverConfig(BaseModel, frozen=True):
    app_token: SecretStr
    user_keys: Annotated[tuple[str, ...], NoDecode]
    emergency_retry_s: int = 30
    emergency_expire_s: int = 600

    @field_validator("user_keys", mode="before")
    @classmethod
    def _keys_from_csv(cls, value: Any) -> Any:
        return _split_csv(value)


class _VisionKeyAlias(PydanticBaseSettingsSource):
    """Reads ANTHROPIC_API_KEY (environment first, then .env) into vision.api_key. It
    sits below the env and dotenv sources, so an explicit BABYMON_VISION__API_KEY wins."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        key = os.environ.get(VISION_KEY_ALIAS) or dotenv_values(ENV_FILE).get(VISION_KEY_ALIAS)
        return {"vision": {"api_key": key}} if key else {}


class Settings(BaseSettings):
    """Env names: BABYMON_CAMERA__USER, BABYMON_VISION__API_KEY,
    BABYMON_TUNING__HEAD_LEFT_ALERT_S, ... ANTHROPIC_API_KEY is also accepted for
    vision.api_key."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=ENV_FILE,
        extra="ignore",
        frozen=True,
    )

    camera: CameraConfig
    vision: VisionConfig | None = None
    pushover: PushoverConfig | None = None
    tuning: Tuning = Tuning()
    data_dir: Path = Path("/data")
    yamnet_path: Path = Path("/models/yamnet.onnx")
    http_port: int = 8080

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, _VisionKeyAlias(settings_cls)

    def capabilities(self) -> Capabilities:
        return Capabilities(vision=self.vision is not None, push=self.pushover is not None)


def env_name(loc: tuple[int | str, ...]) -> str:
    return ENV_PREFIX + "__".join(str(part).upper() for part in loc)


def describe_errors(error: ValidationError) -> list[str]:
    """One line per problem, naming the environment variable to fix. A missing required
    group (the whole camera block) is reported as each variable the group needs."""
    lines: list[str] = []
    for e in error.errors():
        loc = e["loc"]
        group = Settings.model_fields[str(loc[0])].annotation if len(loc) == 1 else None
        if e["type"] == "missing" and isinstance(group, type) and issubclass(group, BaseModel):
            lines.extend(
                f"{env_name((*loc, name))}: {e['msg']}"
                for name, info in group.model_fields.items()
                if info.is_required()
            )
        else:
            lines.append(f"{env_name(loc)}: {e['msg']}")
    return lines


def load() -> Settings:
    """Parse or exit with a message naming each missing or invalid variable."""
    try:
        return Settings()
    except ValidationError as error:
        lines = ["babymon: invalid configuration", *describe_errors(error)]
        print("\n".join(lines), file=sys.stderr)
        raise SystemExit(2) from None
