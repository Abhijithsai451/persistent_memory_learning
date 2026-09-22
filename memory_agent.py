import os
from llm_client import LLMClient
from memory import MemoryManager

class MemoryAgent:

    def __init__(self):
        self.llm = LLMClient(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory_manager = MemoryManager(
            memory_file="benchmark_memory.json"
        )

    def __call__(self, message):
        answer = self.llm.get_response(
            user_message=message,
            memory=self.memory_manager.data
        )
        decision = self.llm.classify_memory(
            user_message=message,
            existing_memories=self.memory_manager.data
        )
        self.memory_manager.save_memory(decision)
        if self.memory_manager.should_consolidate():
            self.memory_manager.consolidate(self.llm)
        return answer

class StatelessMemoryAgent:
    def __init__(self):
        self.llm = LLMClient(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def __call__(self, message):
        prompt = f"""
        You are an AI assistant.

        Answer the user's message directly.

        Do not assume anything about the user
        from previous interactions.

        User message:
        {message}
        """

        response = self.llm.client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        return response.output_text

class NaiveMemoryAgent:
    def __init__(self):
        self.llm = LLMClient(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.memory_manager = MemoryManager(
            memory_file="naive_benchmark_memory.json"
        )
    def __call__(self, message):
        answer = self.llm.get_response(
            user_message=message,
            memory=self.memory_manager.data
        )
        self.memory_manager.add_memory(
            category="conversation",
            content=message
        )

        return answer