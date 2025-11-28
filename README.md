# AI Career Copilot

Interactive Streamlit application that combines LangChain, Ollama, and four bespoke tools to guide ML-focused candidates through career planning:

1. **Skills Gap Analyzer** – compares user skills vs. role requirements and prescribes learning paths.
2. **Resume Scorer** – heuristically scores resumes (0–10) with actionable feedback.
3. **Salary Estimator** – geo + experience-adjusted compensation signals.
4. **Interview Question Generator** – curated technical/behavioral prompts by role and difficulty.

## Prerequisites

- macOS/Linux with Python 3.9+
- [Ollama](https://ollama.com) running locally with at least one supported model pulled (defaults to the lighter `llama3.1:8b`; other options include `llama3.2`, `llama3`, and `phi3:mini`):
  ```bash
  ollama pull llama3.1:8b
  # optionally pull additional models exposed in the sidebar, e.g.
  ollama pull llama3.2
  ollama pull phi3:mini
  ```
- `pip` available (the project uses a virtual environment at `.venv/`).

## Installation

```bash
cd /Users/spartan/Documents/DistributedSystems/Assignment11
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the agent

```bash
source .venv/bin/activate
streamlit run app.py
```

Use the sidebar to set the target role, location, years of experience, skills, and upload a resume (PDF). The chat surface automatically decides when to invoke the four tools.

## Testing

```bash
source .venv/bin/activate
pytest
```

