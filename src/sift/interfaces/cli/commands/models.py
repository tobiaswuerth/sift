from sift.inference.loading import Deferred, deferred
from sift.interfaces.cli.context import Context


def models_of(context: Context) -> Deferred:
    settings = context.settings
    return deferred(settings.embed_model, settings.rerank_model, settings.gpus)
