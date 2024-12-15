import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        HUGGINGFACE_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "huggingface"
        self.name = "huggingface/"

        self.valves = self.Valves(
            **{"HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY", "your-api-key-here")}
        )
        self.url = "https://huggingface.co/api/models"
        self.update_headers()

    def update_headers(self):
        self.headers = {
            "Authorization": f"Bearer {self.valves.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json",
        }

    def get_huggingface_models(self):
        try:
            params = {"limit": 100}  # Nombre de modèles à récupérer par requête
            response = requests.get(self.url, headers=self.headers, params=params)
            response.raise_for_status()
            models = response.json()
            return [
                {"id": model["modelId"], "name": model["modelId"]}
                for model in models
            ]
        except requests.exceptions.RequestException as e:
            return [
                {"id": "error", "name": f"Error fetching models: {str(e)}"}
            ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_headers()

    def pipelines(self) -> List[dict]:
        return self.get_huggingface_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Prepare the payload
            payload = {
                "inputs": user_message,
                "parameters": {
                    "max_new_tokens": body.get("max_tokens", 50),
                    "temperature": body.get("temperature", 0.8),
                    "top_k": body.get("top_k", 40),
                    "top_p": body.get("top_p", 0.9),
                },
            }

            if body.get("stream", False):
                return self.stream_response(model_id, payload)
            else:
                return self.get_completion(model_id, payload)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        try:
            response = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=self.headers, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        yield data.get("generated_text", "")
                    except json.JSONDecodeError:
                        yield "Error parsing JSON response"
        except requests.exceptions.RequestException as e:
            yield f"Error: {e}"

    def get_completion(self, model_id: str, payload: dict) -> str:
        try:
            response = requests.post(f"https://api-inference.huggingface.co/models/{model_id}", headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("generated_text", "")
        except requests.exceptions.RequestException as e:
            return f"Error: {e}"
