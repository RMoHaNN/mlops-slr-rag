"""Sanitized configuration for the MLOps / Industrial AI SLR — RAG project.

This is a copy of the original SLR's screening_config.py with all
domain-specific illustrative examples REMOVED (e.g. named constraint
examples, named deployment strategies, and good/bad coding examples).
Those examples could seed or bias a corpus-grounded system toward
specific findings. Definitions, decision rules, exclusion codes, and
field names are unchanged.

RQ1: What operational constraints distinguish Industrial AI implementation
     (manufacturing/maintenance environments) from Generic AI implementation
     (standard software-centric ML deployment contexts)?
RQ2: What design principles for Industrial AI can be identified from the
     gap between the two literatures?
"""

# =============================================================================
# Models
# =============================================================================

ABSTRACT_SCREENING_MODEL = "gemini-2.5-flash"
ABSTRACT_SCREENING_PROMPT_VERSION = "v1-2026-05-27-sanitized"

FULLTEXT_CODING_MODEL = "gemini-2.5-flash"
FULLTEXT_CODING_PROMPT_VERSION = "v3-2026-06-03-sanitized"

# =============================================================================
# Research questions (verbatim from the original review)
# =============================================================================

RQ1 = (
    "What operational constraints distinguish Industrial AI implementation "
    "(manufacturing/maintenance environments) from Generic AI implementation "
    "(standard software-centric ML deployment contexts)?"
)

RQ2 = (
    "What design principles for Industrial AI can be identified from the "
    "gaps between the two literatures?"
)

# =============================================================================
# Stream definitions (verbatim, minus example lists)
# =============================================================================

STREAM_DEFINITIONS = {
    "industrial": (
        "AI/ML deployment in manufacturing, Industry 4.0, predictive "
        "maintenance, industrial IoT, cyber-physical systems, process "
        "industries, energy/utilities, or safety-critical physical "
        "environments."
    ),
    "generic": (
        "AI/ML deployment in cloud-native, SaaS, enterprise software, data "
        "platforms, or general engineering contexts not tied to physical "
        "production processes."
    ),
}

# =============================================================================
# Coding fields (names and definitions kept; illustrative examples removed)
# =============================================================================

FULLTEXT_CODING_FIELDS = [
    {
        "name": "paper_type",
        "description": (
            "Classify the paper's primary contribution type. Choose one: "
            "architecture | case_study | empirical_study | conceptual | survey."
        ),
    },
    {
        "name": "evidence_level",
        "description": (
            "How grounded is the paper's evidence? Choose one: "
            "real_deployment | prototype | theoretical."
        ),
    },
    {
        "name": "deployment_context",
        "description": (
            "The specific deployment environment described in this paper, "
            "with a one-sentence clarification."
        ),
    },
    {
        "name": "operational_constraints",
        "description": (
            "Constraints and challenges the paper identifies for deploying "
            "or operating AI in this specific context. Extract as a list of "
            "concrete constraints, quoted or closely paraphrased from the "
            "paper. This field directly answers RQ1."
        ),
    },
    {
        "name": "mlops_practices",
        "description": (
            "Specific MLOps or AI operations practices, tools, architectures, "
            "or processes described or proposed in this paper. Name the "
            "practice and how it is applied."
        ),
    },
    {
        "name": "gaps_identified",
        "description": (
            "Gaps in current MLOps/AI operations practice that this paper "
            "explicitly acknowledges — limitations, problems partially "
            "addressed, or statements about what current tools/frameworks "
            "cannot handle. This field feeds into RQ2."
        ),
    },
    {
        "name": "key_findings",
        "description": (
            "Main contribution of the paper in 2 to 4 sentences, paraphrased "
            "from the body, results, and discussion — not the abstract."
        ),
    },
    {
        "name": "method",
        "description": (
            "Research method used, with one sentence on the empirical "
            "grounding (number of cases, sample size, evaluation approach)."
        ),
    },
    {
        "name": "mlops_lifecycle_stage",
        "description": (
            "Which stage(s) of the AI operations lifecycle the paper "
            "primarily addresses."
        ),
    },
    {
        "name": "future_research",
        "description": (
            "Explicit gaps or future directions the authors name, quoted or "
            "closely paraphrased from limitations, conclusion, or "
            "future-work sections."
        ),
    },
]
