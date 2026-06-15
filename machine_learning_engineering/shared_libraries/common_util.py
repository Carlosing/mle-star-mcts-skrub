"""Common utility functions."""

import os
import shutil


def get_text_from_response(response) -> str:
    """Extracts text from an OpenAI-compatible chat completion response."""
    return response.choices[0].message.content


def copy_file(source_file_path: str, destination_dir: str) -> None:
    """Copies a file to the specified directory."""
    if not os.path.isdir(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.copy2(source_file_path, destination_dir)
