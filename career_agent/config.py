"""Centralized configuration for the Career Counseling Agent."""

from __future__ import annotations

APP_NAME = "AI Career Copilot"
APP_DESCRIPTION = (
    "Interactive assistant for AI/ML career planning powered by LangChain, "
    "Ollama, and Streamlit."
)

SUPPORTED_MODELS = [
    "llama3.1:8b",
    "llama3.2",
    "llama3",
    "phi3:mini",
]

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_RESPONSES = 15

TARGET_ROLES = [
    "Machine Learning Engineer",
    "Data Scientist",
    "Deep Learning Engineer",
    "Software Engineer - AI/ML",
]

LOCATIONS = [
    "San Francisco, CA",
    "New York, NY",
    "Austin, TX",
    "Seattle, WA",
    "Remote - US",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are an experienced career coach specializing in AI/ML roles. "
    "Leverage the provided candidate context, resume snippets, and tools "
    "to answer questions with specific, data-backed guidance. "
    "Always reference concrete skills, experience levels, and learning paths. "
    "Use the available tools whenever they can provide structured insights."
)

RESUME_MAX_CHARS = 5500

