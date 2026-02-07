# Generated manually

from django.db import migrations
from django.db import models


class Migration(migrations.Migration):
    dependencies = [
        ("paperless", "0006_applicationconfiguration_barcode_tag_split"),
    ]

    operations = [
        migrations.AlterField(
            model_name="applicationconfiguration",
            name="llm_embedding_backend",
            field=models.CharField(
                blank=True,
                choices=[
                    ("openai", "OpenAI"),
                    ("huggingface", "Huggingface"),
                    ("google_genai", "Google GenAI"),
                ],
                max_length=128,
                null=True,
                verbose_name="Sets the LLM embedding backend",
            ),
        ),
        migrations.AlterField(
            model_name="applicationconfiguration",
            name="llm_backend",
            field=models.CharField(
                blank=True,
                choices=[
                    ("openai", "OpenAI"),
                    ("ollama", "Ollama"),
                    ("gemini", "Gemini"),
                ],
                max_length=128,
                null=True,
                verbose_name="Sets the LLM backend",
            ),
        ),
        migrations.AddField(
            model_name="applicationconfiguration",
            name="llm_ocr_enabled",
            field=models.BooleanField(
                default=False,
                null=True,
                verbose_name="Enables LLM OCR",
            ),
        ),
        migrations.AddField(
            model_name="applicationconfiguration",
            name="llm_ocr_backend",
            field=models.CharField(
                blank=True,
                choices=[
                    ("openai", "OpenAI"),
                    ("gemini", "Gemini"),
                    ("ollama", "Ollama"),
                ],
                max_length=128,
                null=True,
                verbose_name="Sets the LLM OCR backend",
            ),
        ),
        migrations.AddField(
            model_name="applicationconfiguration",
            name="llm_ocr_model",
            field=models.CharField(
                blank=True,
                max_length=128,
                null=True,
                verbose_name="Sets the LLM OCR model",
            ),
        ),
        migrations.AddField(
            model_name="applicationconfiguration",
            name="llm_ocr_api_key",
            field=models.CharField(
                blank=True,
                max_length=1024,
                null=True,
                verbose_name="Sets the LLM OCR API key",
            ),
        ),
        migrations.AddField(
            model_name="applicationconfiguration",
            name="llm_ocr_endpoint",
            field=models.CharField(
                blank=True,
                max_length=256,
                null=True,
                verbose_name="Sets the LLM OCR endpoint, optional",
            ),
        ),
    ]
