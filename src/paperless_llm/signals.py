def get_parser(*args, **kwargs):
    from paperless_llm.parsers import LlmDocumentParser

    return LlmDocumentParser(*args, **kwargs)


def get_supported_mime_types():
    from paperless_llm.parsers import LlmDocumentParser

    return LlmDocumentParser(None).supported_mime_types()


def llm_consumer_declaration(sender, **kwargs):
    return {
        "parser": get_parser,
        "weight": 10,
        "mime_types": get_supported_mime_types(),
    }
