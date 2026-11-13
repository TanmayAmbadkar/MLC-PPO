"""Formatting helpers shared by the benchmark evaluators."""
def format_gold(answer):
    text = str(answer).strip()
    if not text:
        return ''
    if text.startswith(('$', r'\[', r'\(')) or r'\boxed' in text:
        return text
    return '$' + text + '$'
