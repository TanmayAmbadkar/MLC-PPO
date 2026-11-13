"""Binary TRL-compatible math reward; unverifiable golds are rejected explicitly."""
from math_verify import parse, verify, LatexExtractionConfig
from latex2sympy2_extended import NormalizationConfig
from importlib.metadata import version

EXPECTED_VERSIONS = {"math-verify": "0.9.0", "latex2sympy2-extended": "1.11.0", "antlr4-python3-runtime": "4.13.2"}

def require_validated_versions():
    actual = {name: version(name) for name in EXPECTED_VERSIONS}
    if actual != EXPECTED_VERSIONS:
        raise RuntimeError(f"Math verifier versions differ from validation: {actual}; expected {EXPECTED_VERSIONS}")
    return actual

PRED_CONFIG = [LatexExtractionConfig(normalization_config=NormalizationConfig(
    nits=False, malformed_operators=False, basic_latex=True, boxed='all', units=True),
    boxed_match_priority=0, try_extract_without_anchor=False)]

def valid_gold(gold):
    return bool(isinstance(gold,str) and gold.strip() and parse(gold, extraction_mode='first_match', fallback_mode='no_fallback'))

def reward(response, gold):
    if not valid_gold(gold):
        raise ValueError(f'Unparseable gold: {gold!r}')
    parsed_gold = parse(gold, extraction_mode='first_match')
    parsed_response = parse(response, extraction_config=PRED_CONFIG, extraction_mode='first_match')
    return float(verify(parsed_gold, parsed_response))
