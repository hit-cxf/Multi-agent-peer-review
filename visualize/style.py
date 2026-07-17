"""Shared paper-figure colors for all MAPR visualizations."""

COLORS = {
    # Stable semantic colors. Do not remap these meanings between figures.
    "neutral": "#CBD5E1",
    "incorrect": "#D95F59",
    "correct": "#2A9D6F",
    "confidence": "#E9A23B",
    "structure": "#E2E8F0",
    "secondary_text": "#475569",
}

# Answer-transition aliases.
ANSWER_TRANSITION_COLORS = {
    "No Change": COLORS["neutral"],
    "Correct → Incorrect": COLORS["incorrect"],
    "Incorrect → Correct": COLORS["correct"],
    "Incorrect → Incorrect": COLORS["confidence"],
}

# Figure 5 aliases.
CONFIDENCE_FIGURE_COLORS = {
    "correct feedback": COLORS["correct"],
    "incorrect feedback": COLORS["incorrect"],
    "mean confidence": COLORS["confidence"],
    "feedback accuracy": COLORS["correct"],
    "ideal calibration": COLORS["neutral"],
}
