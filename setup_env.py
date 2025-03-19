import os

def init_and_get_env_vars(override_vars=False):
    """
    Initialize and get environment variables.

    Parameters:
    override_vars (bool): If True, override existing environment variables with new values.

    Returns:
    dict: A dictionary containing the environment variables.
    """
    env_vars = {
        "google_cloud_project": "ai-work-in-defect-d01",
        "google_cloud_location": "us-central1",
        "google_genai_use_vertexai": "true",
        "model_name": "gemini-2.0-flash-exp",

    }

    if override_vars:
        for key, value in env_vars.items():
            os.environ[key] = value

    return env_vars