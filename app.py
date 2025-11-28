"""Streamlit front-end for the AI Career Copilot."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import streamlit as st

from career_agent.agent import build_agent
from career_agent.config import (
    APP_DESCRIPTION,
    APP_NAME,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    LOCATIONS,
    SUPPORTED_MODELS,
    TARGET_ROLES,
)
from career_agent.data import CandidateContext
from career_agent.resume_parser import extract_text_from_pdf, extract_text_from_upload

BASE_DIR = Path(__file__).parent
DEFAULT_RESUME_PATH = BASE_DIR / "ShashidharBabuResumeUpdated___ML.pdf"


def _init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "agent_config" not in st.session_state:
        st.session_state.agent_config = {}
    if "default_resume_text" not in st.session_state:
        st.session_state.default_resume_text = _load_default_resume()


def _load_default_resume() -> str:
    if DEFAULT_RESUME_PATH.exists():
        try:
            return extract_text_from_pdf(DEFAULT_RESUME_PATH)
        except Exception as exc:  # pragma: no cover - defensive
            st.warning(f"Could not parse bundled resume: {exc}")
    return ""


def _parse_skill_input(raw: str) -> List[str]:
    if not raw:
        return []
    return sorted({token.strip().lower() for token in raw.split(",") if token.strip()})


def _build_context(form_data: Dict[str, str]) -> CandidateContext:
    skills = _parse_skill_input(form_data["skills"])
    resume_text = form_data["resume"]
    return CandidateContext(
        target_role=form_data["role"],
        skills=skills,
        location=form_data["location"],
        years_experience=form_data["experience"],
        resume_text=resume_text,
        desired_job_description=form_data["job_description"],
    )


def _maybe_rebuild_agent(config_payload: Dict, context: CandidateContext) -> None:
    if st.session_state.agent_config == config_payload and st.session_state.agent_executor:
        return
    if st.session_state.agent_config and st.session_state.agent_config != config_payload:
        st.session_state.chat_history = []
    st.session_state.agent_executor = build_agent(
        context=context,
        model_name=config_payload["model"],
        temperature=config_payload["temperature"],
        system_prompt=config_payload["system_prompt"],
    )
    st.session_state.agent_config = config_payload


def _render_chat() -> None:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about career planning, comps, resumes, etc.")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.agent_executor:
        st.warning("Agent not ready yet. Adjust sidebar settings.")
        return

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            print(f"[Streamlit] Invoking agent with prompt: {prompt}")
            response = st.session_state.agent_executor.invoke({"input": prompt})
            answer = response.get("output", "I could not generate a response.")
            print(f"[Streamlit] Agent response: {answer}")
            st.markdown(answer)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide")
    _init_session_state()

    st.title(APP_NAME)
    st.caption(APP_DESCRIPTION)

    with st.sidebar:
        st.subheader("Configuration")
        model_name = st.selectbox("Ollama Model", SUPPORTED_MODELS, index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, DEFAULT_TEMPERATURE, 0.05)
        target_role = st.selectbox("Target Role", TARGET_ROLES, index=0)
        location = st.selectbox("Location", LOCATIONS, index=0)
        years_experience = st.slider("Years of Experience", 0.0, 20.0, 5.0, 0.5)
        skill_input = st.text_area(
            "Key Skills (comma separated)", value="Python, PyTorch, MLOps"
        )
        job_description = st.text_area(
            "Target JD Snippet (optional)",
            placeholder="Paste bullets from the job you're targeting.",
        )
        resume_upload = st.file_uploader(
            "Upload Resume (PDF)", type=["pdf"], accept_multiple_files=False
        )
        if resume_upload is not None:
            resume_text = extract_text_from_upload(resume_upload)
            st.success(f"{resume_upload.name} uploaded.")
        else:
            resume_text = st.session_state.default_resume_text
            if resume_text:
                st.caption("Using bundled resume for scoring.")
            else:
                st.info("Upload a resume to unlock resume scoring insights.")

        system_prompt = st.text_area(
            "System Instructions",
            value=DEFAULT_SYSTEM_PROMPT,
            help="Advanced: tweak coach tone/policies.",
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skills Gap", "Auto")
    col2.metric("Resume Score", "0–10")
    col3.metric("Salary Range", "Geo-adjusted")
    col4.metric("Interview Prep", "Tech + Behavioral")

    st.divider()
    st.markdown(
        "Chat naturally. The agent will decide when to call the **Skills Gap Analyzer**, "
        "**Resume Scorer**, **Salary Estimator**, or **Interview Question Generator**."
    )

    form_payload = {
        "model": model_name or DEFAULT_MODEL,
        "temperature": temperature,
        "role": target_role,
        "location": location,
        "experience": years_experience,
        "skills": skill_input,
        "resume": resume_text,
        "job_description": job_description,
        "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
    }
    context = _build_context(form_payload)
    _maybe_rebuild_agent(form_payload, context)

    with st.expander("How to get started"):
        st.markdown(
            "- Describe your ideal role, then ask: *What gaps should I close?*\n"
            "- Upload a resume and say: *Score my resume for senior ML engineer.*\n"
            "- Try: *Estimate salary for NYC with 6 years experience.*\n"
            "- Request: *Generate hard behavioral questions for Deep Learning Engineer.*"
        )

    _render_chat()


if __name__ == "__main__":
    main()

