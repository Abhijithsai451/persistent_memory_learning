import os
from llm_client import  LLMClient
from memory import load_memory


def main():
    memory = load_memory()
    # Importing the LLM client
    llm = LLMClient(api_key= os.getenv("OPENAI_API_KEY"))
    while True:
        message = input("Enter your message: ")
        if message.lower() in {"exit", "quit"}:
            break

        answer = llm.get_response(message)
        memory.append({"role": "user", "content": message})
        print(answer)

if __name__ == "__main__":
    main()