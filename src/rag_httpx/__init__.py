"""Agrupa os componentes que implementam o Mini-RAG do HTTPX."""

from .config import SETTINGS, RagSettings  # Expõe a configuração para os demais módulos.

__all__ = ["SETTINGS", "RagSettings"]  # Define os nomes públicos deste pacote.
