"""Smoke tests for the four LangChain tools."""

from __future__ import annotations

import pytest

from career_agent.data import CandidateContext
from career_agent.tools import build_tools


@pytest.fixture
def context() -> CandidateContext:
    return CandidateContext(
        target_role="Machine Learning Engineer",
        skills=["python", "pytorch", "mlops"],
        location="San Francisco, CA",
        years_experience=5.0,
        resume_text="Built ML pipelines in production; led MLOps rollout with PyTorch.",
        desired_job_description="Looking for ML engineer role deploying models.",
    )


@pytest.fixture
def tool_registry(context):
    tools = build_tools(context)
    return {tool.name: tool for tool in tools}


def test_skills_gap_tool(tool_registry):
    tool = tool_registry["skills_gap_analyzer"]
    output = tool.invoke({"user_skills": ["python", "mlops"], "target_role": "Machine Learning Engineer"})
    assert "Skills Gap Analysis" in output
    assert "learning" in output.lower()


def test_resume_scorer(tool_registry):
    tool = tool_registry["resume_scorer"]
    result = tool.invoke({"target_role": "Machine Learning Engineer"})
    assert "/10" in result
    assert "Resume Score" in result


def test_salary_estimator(tool_registry):
    tool = tool_registry["salary_estimator"]
    report = tool.invoke({"location": "New York, NY", "years_experience": 6})
    assert "Salary Signal" in report
    assert "$" in report


def test_interview_question_generator(tool_registry):
    tool = tool_registry["interview_question_generator"]
    questions = tool.invoke(
        {"job_title": "Deep Learning Engineer", "question_type": "technical", "difficulty": "hard", "count": 3}
    )
    assert "Interview Prep" in questions
    assert "1." in questions

