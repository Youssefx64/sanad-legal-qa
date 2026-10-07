"""Tests for settings and model registry configuration."""

from pathlib import Path

import pytest

from sanad.config.registry import (
    ChatModelConfig,
    DefaultsConfig,
    EmbeddingModelConfig,
    ModelRegistry,
    ProviderConfig,
    load_model_registry,
)
from sanad.config.settings import Settings


def test_settings_defaults() -> None:
    """Test default values of application settings."""
    settings = Settings()
    assert settings.environment == "development"
    assert settings.qdrant_url == "http://localhost:6333"
    assert isinstance(settings.cors_origins, list)
    assert "http://localhost:5173" in settings.cors_origins


def test_cors_origins_parsing() -> None:
    """Test parsing comma-separated CORS origins."""
    settings = Settings(CORS_ORIGINS="http://foo.com, http://bar.com")  # type: ignore
    assert settings.cors_origins == ["http://foo.com", "http://bar.com"]


def test_load_real_models_yaml() -> None:
    """Test loading the real config/models.yaml file."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "config" / "models.yaml"
    assert config_path.exists(), f"models.yaml must exist at {config_path}"

    registry = load_model_registry(config_path)
    assert registry.defaults.chat_model in [m.id for m in registry.chat_models]
    assert registry.defaults.embedding_model in [m.id for m in registry.embedding_models]
    assert len(registry.enabled_chat_models) >= 2
    assert len(registry.enabled_embedding_models) >= 2

    # Default lookup
    default_chat = registry.get_chat_model()
    assert default_chat.id == registry.defaults.chat_model

    default_embed = registry.get_embedding_model()
    assert default_embed.id == registry.defaults.embedding_model


def test_registry_duplicate_chat_id() -> None:
    """Verify registry rejects duplicate chat model IDs."""
    with pytest.raises(ValueError, match="Duplicate chat_model id"):
        ModelRegistry(
            defaults=DefaultsConfig(
                chat_model="m1",
                embedding_model="e1",
                judge_model="m1",
            ),
            chat_models=[
                ChatModelConfig(id="m1", label="M1", provider="p", model="mod1"),
                ChatModelConfig(id="m1", label="M1 copy", provider="p", model="mod2"),
            ],
            embedding_models=[
                EmbeddingModelConfig(id="e1", label="E1", provider="p", model="emod1"),
            ],
            providers={"p": ProviderConfig(base_url="http://example.com")},
        )


def test_registry_invalid_default() -> None:
    """Verify registry rejects non-existent default model."""
    with pytest.raises(ValueError, match="Default chat_model 'non_existent' not found"):
        ModelRegistry(
            defaults=DefaultsConfig(
                chat_model="non_existent",
                embedding_model="e1",
                judge_model="non_existent",
            ),
            chat_models=[
                ChatModelConfig(id="m1", label="M1", provider="p", model="mod1"),
            ],
            embedding_models=[
                EmbeddingModelConfig(id="e1", label="E1", provider="p", model="emod1"),
            ],
            providers={"p": ProviderConfig(base_url="http://example.com")},
        )


def test_registry_unknown_provider() -> None:
    """Verify registry rejects enabled model with unregistered provider."""
    with pytest.raises(ValueError, match="specifies unknown provider 'unknown_p'"):
        ModelRegistry(
            defaults=DefaultsConfig(
                chat_model="m1",
                embedding_model="e1",
                judge_model="m1",
            ),
            chat_models=[
                ChatModelConfig(id="m1", label="M1", provider="unknown_p", model="mod1"),
            ],
            embedding_models=[
                EmbeddingModelConfig(id="e1", label="E1", provider="p", model="emod1"),
            ],
            providers={"p": ProviderConfig(base_url="http://example.com")},
        )


def test_disabled_model_lookup_raises() -> None:
    """Verify querying a disabled model raises ValueError."""
    registry = ModelRegistry(
        defaults=DefaultsConfig(
            chat_model="m1",
            embedding_model="e1",
            judge_model="m1",
        ),
        chat_models=[
            ChatModelConfig(id="m1", label="M1", provider="p", model="mod1"),
            ChatModelConfig(id="m2", label="M2", provider="p", model="mod2", enabled=False),
        ],
        embedding_models=[
            EmbeddingModelConfig(id="e1", label="E1", provider="p", model="emod1"),
            EmbeddingModelConfig(id="e2", label="E2", provider="p", model="emod2", enabled=False),
        ],
        providers={"p": ProviderConfig(base_url="http://example.com")},
    )
    with pytest.raises(ValueError, match="is disabled"):
        registry.get_chat_model("m2")

    with pytest.raises(ValueError, match="is disabled"):
        registry.get_embedding_model("e2")

    with pytest.raises(KeyError, match="not found"):
        registry.get_chat_model("unknown_id")

    with pytest.raises(KeyError, match="not found"):
        registry.get_embedding_model("unknown_id")

    # Valid embedding lookup
    embed = registry.get_embedding_model("e1")
    assert embed.id == "e1"

    default_embed = registry.get_embedding_model()
    assert default_embed.id == "e1"


def test_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolving provider base url and api key."""
    provider = ProviderConfig(
        base_url_env="CUSTOM_BASE_URL",
        api_key_env="CUSTOM_KEY",
    )
    assert provider.resolve_base_url() is None
    assert provider.resolve_api_key() is None

    monkeypatch.setenv("CUSTOM_BASE_URL", "http://custom-api.com")
    monkeypatch.setenv("CUSTOM_KEY", "secret-test-key")
    assert provider.resolve_base_url() == "http://custom-api.com"
    assert provider.resolve_api_key() == "secret-test-key"

    # Static base_url overrides env
    provider_static = ProviderConfig(base_url="http://static.com")
    assert provider_static.resolve_base_url() == "http://static.com"


def test_validate_runtime_keys() -> None:
    """Test registry runtime key checking."""
    registry = ModelRegistry(
        defaults=DefaultsConfig(
            chat_model="m1",
            embedding_model="e1",
            judge_model="m1",
        ),
        chat_models=[
            ChatModelConfig(id="m1", label="M1", provider="p", model="mod1"),
        ],
        embedding_models=[
            EmbeddingModelConfig(id="e1", label="E1", provider="p", model="emod1"),
        ],
        providers={"p": ProviderConfig(api_key_env="NON_EXISTENT_KEY_XYZ_123")},
    )
    missing = registry.validate_runtime_keys(allow_missing=True)
    assert len(missing) == 2

    with pytest.raises(RuntimeError, match="Missing API keys"):
        registry.validate_runtime_keys(allow_missing=False)


def test_settings_paths() -> None:
    """Test resolving configuration and data paths."""
    settings = Settings()
    assert settings.get_models_path().exists()
    assert settings.get_rag_path().exists()
    assert settings.get_data_dir().exists()
