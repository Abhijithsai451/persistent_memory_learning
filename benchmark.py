import os
import json

from llm_client import LLMClient
from memory import MemoryManager


# =========================================================
# TEST DATA
# =========================================================

TEST_CASES = [

    {
        "name": "Remember preference",
        "conversation": [
            "I prefer Python for development."
        ],
        "question":
            "What programming language do I prefer?",
        "expected": "Python",
    },

    {
        "name": "Update changed fact",
        "conversation": [
            "I live in Berlin.",
            "I moved to Munich recently."
        ],
        "question":
            "Where do I live?",
        "expected": "Munich",
    },

    {
        "name": "Ignore temporary information",
        "conversation": [
            "I had pizza for lunch today.",
            "I'm building an AI memory system."
        ],
        "question":
            "What long-term project am I working on?",
        "expected": "AI memory system",
    },

    {
        "name": "Ignore temporary location",
        "conversation": [
            "I'm working from a coffee shop today.",
            "My favorite programming language is Python."
        ],
        "question":
            "What is my favorite programming language?",
        "expected": "Python",
    },

    {
        "name": "Changed preference",
        "conversation": [
            "I prefer Python over JavaScript.",
            "Actually, I now prefer Rust over Python."
        ],
        "question":
            "What programming language do I currently prefer?",
        "expected": "Rust",
    },

    {
        "name": "Stable fact among noise",
        "conversation": [
            "I had coffee this morning.",
            "I am building an AI memory research project.",
            "It was raining earlier."
        ],
        "question":
            "What project am I working on?",
        "expected": "AI memory",
    },

    {
        "name": "Multiple preferences",
        "conversation": [
            "I like Python.",
            "I also enjoy hiking."
        ],
        "question":
            "What programming language do I like?",
        "expected": "Python",
    },

    {
        "name": "Repeated information",
        "conversation": [
            "I prefer Python.",
            "I really like using Python.",
            "Python is my favorite language."
        ],
        "question":
            "What language do I prefer?",
        "expected": "Python",
    },
]


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


class ConsolidatedMemoryAgent(
    MemoryAgent
):

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

def run_test_case(agent, test_case):

    print(
        f"\n--- {test_case['name']} ---"
    )

    if hasattr(
        agent,
        "memory_manager"
    ):

        agent.memory_manager.reset()

    for message in test_case[
        "conversation"
    ]:

        agent(message)

    answer = agent(
        test_case["question"]
    )

    passed = agent.llm.evaluate_answer(
        test_case["question"],
        test_case["expected"],
        answer
    )

    if hasattr(
        agent,
        "memory_manager"
    ):

        memory_count = (
            agent.memory_manager
            .get_memory_count()
        )

    else:

        memory_count = 0

    print(
        f"Question: "
        f"{test_case['question']}"
    )

    print(
        f"Expected: "
        f"{test_case['expected']}"
    )

    print(
        f"Actual: {answer}"
    )

    print(
        f"Memory count: "
        f"{memory_count}"
    )

    print(
        f"Result: "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "passed": passed,
        "memory_count": memory_count
    }

def run_agent(agent_class, name):

    print("\n")
    print("=" * 60)
    print( f"BENCHMARK: {name}")
    print("=" * 60)

    agent = agent_class()

    results = []

    for test_case in TEST_CASES:

        result = run_test_case(
            agent,
            test_case
        )

        results.append(result)

    passed = sum(
        result["passed"]
        for result in results
    )

    total = len(results)

    memories = sum(
        result["memory_count"]
        for result in results
    )

    accuracy = (
        passed / total
        if total
        else 0
    )

    print("\n--- Summary ---")

    print(
        f"Agent: {name}"
    )

    print(
        f"Passed: {passed}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.2%}"
    )

    print(
        f"Total memories stored: "
        f"{memories}"
    )

    return {
        "agent": name,
        "passed": passed,
        "total": total,
        "accuracy": accuracy,
        "total_memories": memories
    }

if __name__ == "__main__":

    agents = [
        (
            StatelessAgent,
            "No Memory"
        ),

        (
            NaiveMemoryAgent,
            "Naive Memory"
        ),

        (
            MemoryAgent,
            "Intelligent Memory"
        ),

        (
            ConsolidatedMemoryAgent,
            "Consolidated Memory"
        ),
    ]

    all_results = []

    for agent_class, name in agents:

        result = run_agent(
            agent_class,
            name
        )

        all_results.append(result)

    print("\n")
    print("=" * 60)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 60)

    for result in all_results:

        print(
            f"{result['agent']}: "
            f"{result['accuracy']:.2%} accuracy | "
            f"{result['total_memories']} memories"
        )

    with open(
        "benchmark_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print(
        "\nResults saved to "
        "benchmark_results.json"
    )