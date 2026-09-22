import os

from llm_client import LLMClient
from memory import MemoryManager


class StatelessAgent:
    def __init__(self):
        self.llm = LLMClient(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )
    def __call__(self, message):
        result = self.llm.get_response(
            user_message=message,
            memory_manager=None
        )
        return result["answer"]

class NaiveMemoryAgent:
    def __init__(self):
        self.llm = LLMClient(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )
        self.memory_manager = MemoryManager(
            memory_file=
            "naive_benchmark_memory.json"
        )
    def __call__(self, message):
        result = self.llm.get_response(
            user_message=message,
            memory_manager=self.memory_manager
        )
        self.memory_manager.add_memory(
            category="conversation",
            content=message,
            importance=0.5
        )
        return result["answer"]

class MemoryAgent:
    def __init__(self):

        self.llm = LLMClient(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )

        self.memory_manager = MemoryManager(
            memory_file=
            "intelligent_benchmark_memory.json"
        )

    def __call__(self, message):

        result = self.llm.get_response(
            user_message=message,
            memory_manager=self.memory_manager
        )

        decision = self.llm.classify_memory(
            user_message=message,
            existing_memories=
                self.memory_manager.data
        )

        self.memory_manager.save_memory(
            decision
        )

        return result["answer"]


class ConsolidatedMemoryAgent(MemoryAgent):

    def __call__(self, message):

        result = self.llm.get_response(
            user_message=message,
            memory_manager=self.memory_manager
        )

        decision = self.llm.classify_memory(
            user_message=message,
            existing_memories=
                self.memory_manager.data
        )

        self.memory_manager.save_memory(
            decision
        )

        if self.memory_manager.should_consolidate(
            threshold=5
        ):
            self.memory_manager.consolidate(
                self.llm
            )
        return result["answer"]