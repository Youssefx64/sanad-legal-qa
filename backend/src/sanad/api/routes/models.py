"""Model catalog inspection endpoint."""

from fastapi import APIRouter, Depends

from sanad.api.deps import get_model_registry_dep
from sanad.api.schemas import ModelsResponse
from sanad.config.registry import ModelRegistry

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=ModelsResponse)
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry_dep),
) -> ModelsResponse:
    """List registered and enabled chat models, embedding models, and default selections."""
    chat_list = [
        {
            "id": m.id,
            "label": m.label,
            "provider": m.provider,
            "model": m.model,
            "supports_streaming": m.supports_streaming,
            "enabled": m.enabled,
        }
        for m in registry.enabled_chat_models
    ]

    embed_list = [
        {
            "id": m.id,
            "label": m.label,
            "provider": m.provider,
            "model": m.model,
            "dimension": m.dimension,
            "enabled": m.enabled,
        }
        for m in registry.enabled_embedding_models
    ]

    defaults = {
        "chat_model": registry.defaults.chat_model,
        "embedding_model": registry.defaults.embedding_model,
        "judge_model": registry.defaults.judge_model,
    }

    return ModelsResponse(
        defaults=defaults,
        chat_models=chat_list,
        embedding_models=embed_list,
    )

