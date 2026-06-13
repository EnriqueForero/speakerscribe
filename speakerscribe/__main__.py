"""Module entry point: enables ``python -m speakerscribe``.

Equivalent to the ``speakerscribe`` console script — useful when the
script shims are not on PATH (some Colab/venv setups).
"""

from speakerscribe.cli import app

if __name__ == "__main__":
    app()
