import requests

def check_local_llm_models(base_url="http://localhost:1234"):
    """Check available models from a local LLM instance."""
    endpoint = f"{base_url}/v1/models"
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        data = response.json()
        print("Available models:")
        for model in data.get("data", []):
            print("-", model.get("id"))
    except requests.exceptions.RequestException as e:
        print("Error checking LLM models:", e)

if __name__ == "__main__":
    check_local_llm_models()
