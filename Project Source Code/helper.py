import os


def load_options(filename):
    """Helper function to load options from a text file"""
    file_path = os.path.join('static', 'input_fields', filename)
    try:
        with open(file_path, 'r') as f:
            options = [line.strip() for line in f if line.strip()]
        return sorted(list(set(options)))  # Remove duplicates and sort
    except FileNotFoundError:
        return []  # Return empty list if file not found