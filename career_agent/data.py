"""Domain data powering the Career Counseling Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

JOB_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "Machine Learning Engineer": {
        "summary": (
            "Builds end-to-end ML systems, deploying models into production with "
            "robust monitoring, MLOps, and CI/CD practices."
        ),
        "core_skills": [
            "python",
            "pytorch",
            "tensorflow",
            "feature engineering",
            "mlops",
            "data pipelines",
            "model optimization",
            "distributed training",
        ],
        "tooling": [
            "aws sagemaker",
            "mlflow",
            "kubernetes",
            "docker",
            "airflow",
        ],
        "learning_focus": [
            "Experiment tracking and ML observability",
            "Model deployment patterns on cloud providers",
            "Designing resilient data + feature stores",
        ],
    },
    "Data Scientist": {
        "summary": (
            "Transforms raw data into insights using statistical modeling, "
            "experimentation, and stakeholder storytelling."
        ),
        "core_skills": [
            "python",
            "sql",
            "statistics",
            "experimentation",
            "bayesian modeling",
            "feature engineering",
            "data visualization",
        ],
        "tooling": [
            "dbt",
            "snowflake",
            "powerbi",
            "tableau",
        ],
        "learning_focus": [
            "Experiment design for ambiguous business problems",
            "Communicating impact with product partners",
        ],
    },
    "Deep Learning Engineer": {
        "summary": "Owns research-to-production workflows for deep neural networks.",
        "core_skills": [
            "python",
            "pytorch",
            "tensorboard",
            "distributed training",
            "model compression",
            "cuda",
            "transformers",
            "evaluation pipelines",
        ],
        "tooling": [
            "ray",
            "onnx",
            "huggingface",
            "deepspeed",
            "triton",
        ],
        "learning_focus": [
            "Advanced optimization (LoRA, QLoRA, pruning)",
            "Latency-aware serving for LLMs/CV models",
        ],
    },
    "Software Engineer - AI/ML": {
        "summary": (
            "Ships user-facing experiences that embed ML models inside large-scale "
            "software systems with strong engineering best practices."
        ),
        "core_skills": [
            "python",
            "java",
            "api design",
            "distributed systems",
            "feature stores",
            "monitoring",
            "model integration",
        ],
        "tooling": [
            "grpc",
            "kafka",
            "aws",
            "gcp",
            "kubernetes",
        ],
        "learning_focus": [
            "Latency budgets and SLAs for ML services",
            "Cross-team collaboration for experimentation",
        ],
    },
}

LEARNING_RESOURCES: Dict[str, List[Dict[str, str]]] = {
    "mlops": [
        {
            "title": "Full Stack Deep Learning - MLOps",
            "provider": "FSDL",
            "url": "https://fullstackdeeplearning.com",
        },
        {
            "title": "Practical MLOps",
            "provider": "O'Reilly",
            "url": "https://www.oreilly.com/library/view/practical-mlops",
        },
    ],
    "distributed training": [
        {
            "title": "Scaling PyTorch Training",
            "provider": "Meta AI",
            "url": "https://pytorch.org/tutorials/",
        }
    ],
    "transformers": [
        {
            "title": "Hugging Face Transformers Course",
            "provider": "Hugging Face",
            "url": "https://huggingface.co/learn",
        }
    ],
    "statistics": [
        {
            "title": "Experimentation & Causal Inference",
            "provider": "Udacity",
            "url": "https://www.udacity.com/course/causal-inference",
        }
    ],
    "feature engineering": [
        {
            "title": "Feature Engineering for ML",
            "provider": "Coursera",
            "url": "https://www.coursera.org/learn/feature-engineering",
        }
    ],
    "api design": [
        {
            "title": "Designing ML APIs with FastAPI",
            "provider": "TestDriven.io",
            "url": "https://testdriven.io/courses/fastapi-machine-learning/",
        }
    ],
}

SALARY_BASELINES = {
    "Machine Learning Engineer": 175000,
    "Data Scientist": 150000,
    "Deep Learning Engineer": 185000,
    "Software Engineer - AI/ML": 180000,
}

LOCATION_FACTORS = {
    "San Francisco, CA": 1.22,
    "New York, NY": 1.18,
    "Seattle, WA": 1.12,
    "Austin, TX": 1.0,
    "Remote - US": 0.95,
}

LOCATION_ALIASES = {
    "san francisco": "San Francisco, CA",
    "sf": "San Francisco, CA",
    "bay area": "San Francisco, CA",
    "new york": "New York, NY",
    "nyc": "New York, NY",
    "new york city": "New York, NY",
    "seattle": "Seattle, WA",
    "austin": "Austin, TX",
    "remote": "Remote - US",
}

EXPERIENCE_FACTORS: List[Tuple[int, int, float]] = [
    (0, 2, 0.85),
    (2, 5, 1.0),
    (5, 8, 1.15),
    (8, 50, 1.32),
]

QUESTION_BANK = {
    "Machine Learning Engineer": {
        "technical": {
            "easy": [
                "How would you explain bias-variance tradeoff to a junior teammate?",
                "Walk me through your approach to monitoring a batch training pipeline.",
            ],
            "medium": [
                "Describe how you would productionize a gradient boosting model "
                "that currently lives in a notebook.",
                "What are the trade-offs between Feast and homegrown feature stores?",
            ],
            "hard": [
                "Design a multi-tenant training service that auto-scales on Kubernetes.",
                "How would you implement automated rollback for champion/challenger "
                "models served via SageMaker Endpoints?",
            ],
        },
        "behavioral": {
            "easy": [
                "Tell me about a time you simplified an ML system for maintainability."
            ],
            "medium": [
                "Describe a disagreement with data science partners and how you resolved it."
            ],
            "hard": [
                "Share a time you had to sunset a successful ML product. What happened?"
            ],
        },
    },
    "Data Scientist": {
        "technical": {
            "easy": [
                "Explain the intuition behind logistic regression.",
                "How do you handle missing data before running an experiment?",
            ],
            "medium": [
                "Design an uplift experiment for a churn-prevention campaign.",
                "When would you prefer Bayesian A/B testing over frequentist methods?",
            ],
            "hard": [
                "Walk through building a causal model for marketplace supply-demand balance.",
            ],
        },
        "behavioral": {
            "easy": [
                "Describe a dashboard you built that changed stakeholder behavior."
            ],
            "medium": [
                "Talk about a time data contradicted executive intuition."
            ],
            "hard": [
                "Share how you navigated delivering negative findings to leadership."
            ],
        },
    },
    "Deep Learning Engineer": {
        "technical": {
            "easy": [
                "Compare CNNs and Transformers for vision tasks.",
                "What is gradient checkpointing and why is it useful?",
            ],
            "medium": [
                "How would you optimize inference latency for a 70B parameter LLM?",
                "Discuss techniques to stabilize GAN training.",
            ],
            "hard": [
                "Design a retrieval-augmented generation (RAG) stack optimized for on-call debugging.",
            ],
        },
        "behavioral": {
            "easy": [
                "Describe a time you translated research findings into product value."
            ],
            "medium": [
                "Tell me about mentoring someone ramping up on transformers."
            ],
            "hard": [
                "Share a time you aligned research and platform teams on shared OKRs."
            ],
        },
    },
    "Software Engineer - AI/ML": {
        "technical": {
            "easy": [
                "How do you expose a model-inference service safely via REST/GRPC?",
            ],
            "medium": [
                "Design a feature flag system for rolling out ML-backed product changes.",
            ],
            "hard": [
                "Architect a low-latency streaming inference stack handling 50k RPS.",
            ],
        },
        "behavioral": {
            "easy": [
                "Talk about partnering with product to define ML success metrics."
            ],
            "medium": [
                "Tell me about building alignment between platform and product teams."
            ],
            "hard": [
                "Describe a time you defended engineering quality under tight deadlines."
            ],
        },
    },
}


@dataclass
class CandidateContext:
    """Runtime context shared with LangChain tools."""

    target_role: str
    skills: List[str]
    location: str
    years_experience: float
    resume_text: str
    desired_job_description: str = ""


