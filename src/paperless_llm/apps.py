from django.apps import AppConfig

from paperless_llm.signals import llm_consumer_declaration


class PaperlessLlmConfig(AppConfig):
    name = "paperless_llm"

    def ready(self) -> None:
        from documents.signals import document_consumer_declaration

        document_consumer_declaration.connect(llm_consumer_declaration)

        AppConfig.ready(self)
