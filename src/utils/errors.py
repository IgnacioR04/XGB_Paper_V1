"""Tipos de error usados por el bot."""


class PaperTraderError(Exception):
    pass


class DataFetchError(PaperTraderError):
    pass


class StaleDataError(PaperTraderError):
    pass


class FeatureSchemaMismatch(PaperTraderError):
    pass


class ArtifactMissing(PaperTraderError):
    pass
