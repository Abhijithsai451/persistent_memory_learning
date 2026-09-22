import os

from llm_client import LLMClient
from memory import MemoryManager


def main():
    memory_manager = MemoryManager()
    llm = LLMClient(api_key=os.getenv("OPENAI_API_KEY"))

    print("AI Memory Agent")

    print("Type 'exit' to quit." )

    while True:
        message = input("\nYou: ")

        if message.lower() in {"exit","quit"}:
            break

        result = llm.get_response(
            user_message=message,
            memory_manager=memory_manager
        )

        print(f"\nAgent: {result['answer']}")

        decision = llm.classify_memory(
            user_message=message,
            existing_memories=
                memory_manager.data
        )
        memory_manager.save_memory(decision)

        if memory_manager.should_consolidate(threshold=5):
            memory_manager.consolidate(llm)


if __name__ == "__main__":
    main()