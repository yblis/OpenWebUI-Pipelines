import os
from transformers import pipeline
from huggingface_hub import login
from transformers.pipelines import PipelineException


class HuggingFaceChatPipeline:
    def __init__(self, api_key: str, model_id: str):
        """
        Initialisation de la classe pour interagir avec un modèle Hugging Face.

        :param api_key: Clé API Hugging Face.
        :param model_id: Identifiant du modèle à utiliser.
        """
        self.api_key = api_key
        self.model_id = model_id
        self.pipeline = None

    def authenticate(self):
        """
        Authentifie auprès de Hugging Face en utilisant la clé API.
        """
        try:
            login(token=self.api_key)
            print("Authentification réussie.")
        except Exception as e:
            print(f"Erreur d'authentification : {e}")
            raise

    def initialize_pipeline(self):
        """
        Initialise le pipeline pour interagir avec le modèle spécifié.
        """
        try:
            self.pipeline = pipeline("text-generation", model=self.model_id, use_auth_token=self.api_key)
            print(f"Pipeline initialisé avec succès pour le modèle {self.model_id}.")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du pipeline : {e}")
            raise

    def chat(self, user_input: str, max_tokens: int = 50, temperature: float = 0.8) -> str:
        """
        Envoie une requête au modèle et retourne la réponse générée.

        :param user_input: Texte d'entrée de l'utilisateur.
        :param max_tokens: Nombre maximum de tokens pour la réponse.
        :param temperature: Température pour contrôler la créativité de la réponse.
        :return: Texte généré par le modèle.
        """
        if not self.pipeline:
            print("Pipeline non initialisé. Appelez `initialize_pipeline` d'abord.")
            return "Erreur : Pipeline non initialisé."

        try:
            response = self.pipeline(
                user_input,
                max_length=max_tokens,
                temperature=temperature,
                return_full_text=False
            )
            return response[0]["generated_text"]
        except PipelineException as e:
            print(f"Erreur dans le pipeline : {e}")
            return "Erreur : Impossible de générer une réponse."
        except Exception as e:
            print(f"Une erreur est survenue : {e}")
            return "Erreur : Problème inconnu."

    def test_chat(self):
        """
        Teste le pipeline avec un exemple de requête.
        """
        user_input = "Bonjour, comment allez-vous ?"
        print("Utilisateur :", user_input)
        response = self.chat(user_input)
        print("Modèle :", response)


if __name__ == "__main__":
    # Remplacez par votre clé API Hugging Face
    API_KEY = os.getenv("HUGGINGFACE_API_KEY", "votre_clé_api")
    MODEL_ID = "Qwen/QwQ-32B-Preview"  # Exemple de modèle

    # Initialisation de la classe
    chat_pipeline = HuggingFaceChatPipeline(api_key=API_KEY, model_id=MODEL_ID)

    # Authentification auprès de Hugging Face
    try:
        chat_pipeline.authenticate()
    except Exception:
        print("Impossible de s'authentifier.")
        exit()

    # Initialisation du pipeline
    try:
        chat_pipeline.initialize_pipeline()
    except Exception:
        print("Impossible d'initialiser le pipeline.")
        exit()

    # Test de la conversation
    chat_pipeline.test_chat()
