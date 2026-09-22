import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
class LLMClient:
    def __init__(self, api_key) -> None:
        self.api_key = api_key
        self.client = None

    def get_llm_client(self):
        self.client = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        return self.client

    def get_response(self, user_message: str, memory: list)-> str:
        context = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in memory
        )

        prompt = f"""
                Here is the previous Conversation: 
                {context}.
                Current user message: {user_message}   
                """

        client = self.get_llm_client()
        response = client.responses.create(
            model = "gpt-5-mini",
            input = prompt
        )
        return response.output_text
