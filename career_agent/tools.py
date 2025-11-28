"""LangChain tools backing the Career Counseling Agent."""

from __future__ import annotations

import difflib
import random
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from .data import (
    CandidateContext,
    EXPERIENCE_FACTORS,
    JOB_PROFILES,
    LEARNING_RESOURCES,
    LOCATION_FACTORS,
    LOCATION_ALIASES,
    QUESTION_BANK,
    SALARY_BASELINES,
)

ROLE_ALIASES = {
    "ml engineer": "Machine Learning Engineer",
    "machine learning": "Machine Learning Engineer",
    "data scientist": "Data Scientist",
    "deep learning": "Deep Learning Engineer",
    "deep learning engineer": "Deep Learning Engineer",
    "software engineer ai": "Software Engineer - AI/ML",
    "software engineer - ai/ml": "Software Engineer - AI/ML",
}


class SkillsGapInput(BaseModel):
    user_skills: Optional[Union[List[str], str]] = Field(
        default=None,
        description="List of skills the candidate currently possesses.",
    )
    target_role: Optional[str] = Field(
        default=None,
        description="Role to compare against (default: session target role).",
    )
    job_description: Optional[str] = Field(
        default=None,
        description="Optional JD snippet for additional context.",
    )
    years_experience: Optional[float] = Field(
        default=None,
        description="Approximate years of experience for leveling guidance.",
    )


class ResumeScorerInput(BaseModel):
    resume_text: Optional[str] = Field(
        default=None,
        description="Raw resume text or summary. Defaults to uploaded resume.",
    )
    target_role: Optional[str] = Field(
        default=None, description="Role to optimize for (default: session target role)."
    )


class SalaryEstimatorInput(BaseModel):
    job_title: Optional[str] = Field(default=None, description="Desired role title.")
    location: Optional[str] = Field(
        default=None, description="City/region (default: session location)."
    )
    years_experience: Optional[float] = Field(
        default=None, description="Years of relevant experience."
    )


class InterviewQuestionInput(BaseModel):
    job_title: Optional[str] = Field(default=None, description="Target job title.")
    question_type: str = Field(
        default="mixed", description="technical, behavioral, or mixed."
    )
    difficulty: str = Field(
        default="medium", description="easy, medium, or hard difficulty."
    )
    count: int = Field(
        default=5, ge=1, le=10, description="Number of questions to generate."
    )


def build_tools(context: CandidateContext) -> List[StructuredTool]:
    """Create LangChain tools bound to the current candidate context."""

    def skills_gap_analyzer(
        user_skills: Optional[List[str]] = None,
        target_role: Optional[str] = None,
        job_description: Optional[str] = None,
        years_experience: Optional[float] = None,
    ) -> str:
        role = _resolve_role(target_role or context.target_role)
        profile = JOB_PROFILES.get(role)
        if profile is None:
            return f"No profile found for {role}. Choose one of: {', '.join(JOB_PROFILES)}"

        parsed_skills = _normalize_skills(user_skills) or context.skills
        jd_skills = _extract_keywords(job_description or context.desired_job_description)
        combined = {s.lower() for s in parsed_skills}.union(jd_skills)

        required = set(map(str.lower, profile["core_skills"] + profile["tooling"]))
        matched = sorted(required.intersection(combined))
        gaps = sorted(required.difference(combined))
        leverage = sorted(set(combined) - required)

        learning_recs = _learning_plan(gaps)
        exp = years_experience if years_experience is not None else context.years_experience
        leveling_note = _leveling_guidance(exp)

        lines = [
            f"### Skills Gap Analysis → {role}",
            f"- Experience reference: ~{exp:.1f} years",
            f"- Matched strengths ({len(matched)}): {', '.join(matched) or 'None captured'}",
            f"- Priority gaps ({len(gaps)}): {', '.join(gaps) or 'All core skills covered'}",
        ]
        if leverage:
            lines.append(f"- Transferable extras: {', '.join(leverage)}")
        if learning_recs:
            lines.append("\n**Suggested Learning Path:**")
            for skill, recs in learning_recs.items():
                lines.append(f"- {skill.title()}: " + "; ".join(f"{r['title']} ({r['provider']})" for r in recs))
        lines.append("\n**Leveling Tip:** " + leveling_note)
        return "\n".join(lines)

    def resume_scorer(
        resume_text: Optional[str] = None,
        target_role: Optional[str] = None,
    ) -> str:
        role = _resolve_role(target_role or context.target_role)
        profile = JOB_PROFILES.get(role)
        if profile is None:
            return f"Unknown role '{role}'."

        text = (resume_text or context.resume_text or "").strip()
        if not text:
            return "No resume content supplied. Upload a resume or paste plain text."

        lower = text.lower()
        feedback = []
        score = 4.0

        # Skill coverage
        matched = [skill for skill in profile["core_skills"] if skill in lower]
        coverage_ratio = len(matched) / len(profile["core_skills"])
        score += coverage_ratio * 3.0
        if coverage_ratio < 0.6:
            missing = [skill for skill in profile["core_skills"] if skill not in matched]
            feedback.append(
                f"Highlight missing fundamentals: {', '.join(missing[:5])}."
            )

        # Quantified impact
        quant_hits = sum(lower.count(k) for k in ["%", "$", "x", "reduced", "improved"])
        if quant_hits >= 8:
            score += 1.2
        elif quant_hits >= 3:
            score += 0.6
        else:
            feedback.append("Add quantified impact metrics (%, $, latency, users).")

        # Leadership / collaboration
        leadership_terms = sum(
            lower.count(term)
            for term in ["led", "mentored", "cross-functional", "stakeholder", "partnered"]
        )
        if leadership_terms:
            score += 0.4
        else:
            feedback.append("Include collaboration stories (e.g., partnered with product, led MLOps rollout).")

        # MLOps / production keywords
        prod_terms = sum(
            lower.count(term) for term in ["deployed", "pipeline", "mlops", "monitor"]
        )
        if prod_terms:
            score += 0.6
        else:
            feedback.append("Call out production hardening (monitoring, rollbacks, SLAs).")

        # Length / structure
        word_count = len(text.split())
        if word_count < 250:
            feedback.append("Expand experience bullets with more context (resume < 1 page).")
        elif word_count > 1400:
            feedback.append("Condense to 1-2 pages; recruiters skim quickly.")
        else:
            score += 0.2

        final_score = max(0.0, min(10.0, round(score, 1)))
        lines = [
            f"### Resume Score ({role}) → **{final_score}/10**",
            f"- Core skill coverage: {coverage_ratio:.0%}",
            f"- Impact statements detected: {quant_hits}",
            f"- Leadership signals: {leadership_terms}",
        ]
        if feedback:
            lines.append("\n**Actionable Improvements**")
            for tip in feedback:
                lines.append(f"- {tip}")
        else:
            lines.append("\nResume already aligns strongly with the target role—focus on tailoring summary statements.")
        return "\n".join(lines)

    def salary_estimator(
        job_title: Optional[str] = None,
        location: Optional[str] = None,
        years_experience: Optional[float] = None,
    ) -> str:
        role = _resolve_role(job_title or context.target_role)
        base = SALARY_BASELINES.get(role)
        if base is None:
            return f"Salary data unavailable for {role}."

        loc = _normalize_location(location) or context.location
        loc_factor = LOCATION_FACTORS.get(loc, 1.0)
        yrs = years_experience if years_experience is not None else context.years_experience
        exp_factor = _experience_multiplier(yrs)
        median = base * loc_factor * exp_factor
        low = round(median * 0.9 / 1000) * 1000
        high = round(median * 1.15 / 1000) * 1000

        lines = [
            f"### Salary Signal for {role}",
            f"- Location: {loc} (market factor ×{loc_factor:.2f})",
            f"- Experience: {yrs:.1f} yrs (level factor ×{exp_factor:.2f})",
            f"- Estimated range: **${low:,.0f} – ${high:,.0f}**",
            "These numbers blend aggregated public reports, compensation ladders, and cost-of-living adjustments.",
            "Use the high end when you demonstrate production ownership + leadership; the low end fits growth candidates.",
        ]
        return "\n".join(lines)

    def interview_question_generator(
        job_title: Optional[str] = None,
        question_type: str = "mixed",
        difficulty: str = "medium",
        count: int = 5,
    ) -> str:
        role = _resolve_role(job_title or context.target_role)
        bank = QUESTION_BANK.get(role)
        if bank is None:
            return f"No interview bank available for {role}."

        question_type = question_type.lower()
        difficulty = difficulty.lower()
        choices = []

        types_to_use = (
            ["technical", "behavioral"] if question_type == "mixed" else [question_type]
        )
        for qtype in types_to_use:
            difficulty_key = bank.get(qtype, {})
            bucket = difficulty_key.get(difficulty) or next(
                iter(difficulty_key.values()), []
            )
            choices.extend((qtype, q) for q in bucket)

        if not choices:
            return f"No questions found for type={question_type}, difficulty={difficulty}."

        random.shuffle(choices)
        selected = choices[:count]
        lines = [f"### {role} Interview Prep ({difficulty.title()} / {question_type.title()})"]
        for idx, (qtype, question) in enumerate(selected, 1):
            lines.append(f"{idx}. ({qtype[:4]}) {question}")
        lines.append("\nUse the STAR(L) framework and tie answers back to metrics + learning.")
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            name="skills_gap_analyzer",
            func=skills_gap_analyzer,
            description=(
                "Compare candidate skills with target job requirements, highlight gaps, "
                "and suggest learning resources."
            ),
            args_schema=SkillsGapInput,
        ),
        StructuredTool.from_function(
            name="resume_scorer",
            func=resume_scorer,
            description=(
                "Score a resume (0-10) for a target role and provide actionable feedback."
            ),
            args_schema=ResumeScorerInput,
        ),
        StructuredTool.from_function(
            name="salary_estimator",
            func=salary_estimator,
            description=(
                "Estimate realistic salary ranges based on role, location, and experience."
            ),
            args_schema=SalaryEstimatorInput,
        ),
        StructuredTool.from_function(
            name="interview_question_generator",
            func=interview_question_generator,
            description=(
                "Produce tailored technical/behavioral interview questions by role and difficulty."
            ),
            args_schema=InterviewQuestionInput,
        ),
    ]


def _resolve_role(role: Optional[Union[str, Dict[str, Any]]]) -> str:
    if not role:
        return "Machine Learning Engineer"
    if isinstance(role, dict):
        role = role.get("job_title") or role.get("role") or ""
    elif isinstance(role, str) and role.strip().startswith("{"):
        try:
            parsed = json.loads(role)
            if isinstance(parsed, dict):
                role = parsed.get("job_title") or parsed.get("role") or role
        except json.JSONDecodeError:
            pass
    role_lower = role.lower().strip()
    if role_lower in ROLE_ALIASES:
        return ROLE_ALIASES[role_lower]
    match = difflib.get_close_matches(role_lower, [r.lower() for r in JOB_PROFILES], n=1)
    if match:
        idx = [r.lower() for r in JOB_PROFILES].index(match[0])
        return list(JOB_PROFILES.keys())[idx]
    return role.title()


def _normalize_skills(skills: Optional[Union[Iterable[str], str]]) -> List[str]:
    if not skills:
        return []
    if isinstance(skills, str):
        skills = [skills]
    parsed = []
    for skill in skills:
        if not skill:
            continue
        parsed.extend([token.strip().lower() for token in skill.split(",") if token.strip()])
    return sorted(set(parsed))


def _extract_keywords(text: Optional[str]) -> set:
    if not text:
        return set()
    tokens = {
        token.strip(" ,.;:()").lower()
        for token in text.split()
        if len(token) > 2
    }
    return {t for t in tokens if t.isalpha()}


def _learning_plan(missing: Sequence[str]) -> Dict[str, List[Dict[str, str]]]:
    plan: Dict[str, List[Dict[str, str]]] = {}
    for skill in missing:
        if skill in LEARNING_RESOURCES:
            plan[skill] = LEARNING_RESOURCES[skill][:2]
    return plan


def _leveling_guidance(years_experience: float) -> str:
    if years_experience < 3:
        return "Position yourself as a high-growth IC1/IC2 highlighting shipped projects and rapid learning."
    if years_experience < 6:
        return "Target mid-level roles (IC3) emphasizing autonomous delivery and MLOps contributions."
    if years_experience < 9:
        return "Lean into senior scope: cross-functional leadership, system design, and mentoring."
    return "Highlight staff-level impact: roadmaps, platform ownership, and influencing exec decisions."


def _experience_multiplier(years: float) -> float:
    for lower, upper, factor in EXPERIENCE_FACTORS:
        if lower <= years < upper:
            return factor
    return EXPERIENCE_FACTORS[-1][2]


def _normalize_location(location: Optional[Union[str, Dict[str, Any]]]) -> Optional[str]:
    if not location:
        return None
    if isinstance(location, dict):
        location = location.get("location") or location.get("city") or ""
    elif isinstance(location, str) and location.strip().startswith("{"):
        try:
            parsed = json.loads(location)
            if isinstance(parsed, dict):
                location = parsed.get("location") or location.get("city") or location
        except json.JSONDecodeError:
            pass
    candidate = location.strip()
    if candidate in LOCATION_FACTORS:
        return candidate
    key = candidate.lower()
    if key in LOCATION_ALIASES:
        return LOCATION_ALIASES[key]
    for loc in LOCATION_FACTORS:
        if key == loc.lower():
            return loc
    return candidate
