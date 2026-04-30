"""Safety guardrail: detect and block medical-advice requests."""
import re

MEDICAL_ADVICE_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bshould i (take|use|try|start|stop|increase|decrease|switch)\b",
        r"\bcan i (take|use|combine|mix)\b",
        r"\b(what|which) (dose|dosage|mg|milligram|pill|tablet|medication|drug) (should|can|to)\b",
        r"\b(what|recommended|standard|typical|appropriate) dosage.*(patient|person|people|individual|i |me )\b",
        r"\bdosage.*(recommended for|appropriate for|should|take|i should)\b",
        r"\b(how much|how many) (mg|milligrams|ml|pills|tablets|drops)\b",
        r"\btreat (my|me|myself|our)\b",
        r"\bdiagnos(e|is|ing) (my|me|myself)\b",
        r"\b(cure|curing|treating) my\b",
        r"\bmedical advice\b",
        r"\bis it safe (for me|to take|to use)\b",
        r"\bside effect.*i (should|might|will|could)\b",
        r"\bam i (sick|ill|having|suffering)\b",
        r"\bdo i have\b",
        r"\bmy (symptoms|condition|disease|illness|pain)\b",
        r"\bprescri(be|ption) (for me|me)\b",
        r"\bself[- ]medic\b",
    ]
]

_SAFE_REFUSAL = (
    "I'm a pharmaceutical **research** assistant designed for R&D data exploration — "
    "I cannot provide medical advice, diagnosis, or treatment recommendations for individuals. "
    "Please consult a qualified healthcare professional for any personal health questions."
)


def is_medical_advice(query: str) -> bool:
    """Return True if the query appears to be requesting personal medical advice."""
    return any(p.search(query) for p in MEDICAL_ADVICE_PATTERNS)


def safe_refusal_message() -> str:
    return _SAFE_REFUSAL
