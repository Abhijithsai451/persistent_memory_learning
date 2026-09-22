import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMClient:

    def __init__(self, api_key=None):

        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
        )

        self.client = OpenAI(
            api_key=self.api_key
        )

    def get_response(
        self,
        user_message,
        memory_manager=None
    ):

        start_time = time.perf_counter()

        if memory_manager:

            relevant_memories = (
                memory_manager.search_memory(
                    user_message
                )
            )

        else:
            relevant_memories = []

        context = "\n".join(
            f"- [{item['category']}] "
            f"{item['content']}"
            for item in relevant_memories
        )

        if not context:
            context = "No relevant memories found."

        prompt = f"""
                You are an AI assistant with long-term memory.

                Relevant information about the user:

                {context}
                
                Use this information when answering the user.
                
                Rules:
                
                - Treat relevant stored memories as known information.
                - Do not ask the user to confirm a memory unless
                  they explicitly indicate it may be wrong.
                - Answer directly and naturally.
                - Do not mention the internal memory system.
                - Do not provide unsolicited recommendations.
                - Do not invent personal information.
                
                User message:
                
                {user_message}
                """

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        latency = (
            time.perf_counter()
            - start_time
        )

        usage = getattr(
            response,
            "usage",
            None
        )

        input_tokens = (
            getattr(
                usage,
                "input_tokens",
                0
            )
            if usage
            else 0
        )

        output_tokens = (
            getattr(
                usage,
                "output_tokens",
                0
            )
            if usage
            else 0
        )

        return {
            "answer": response.output_text,
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "retrieved_memories": len(
                relevant_memories
            )
        }

    # -------------------------
    # Memory classification
    # -------------------------

    def classify_memory(
        self,
        user_message,
        existing_memories
    ):

        prompt = f"""
                You manage long-term memory for an AI assistant.
                
                Decide whether the following user message contains
                information that should be remembered for future conversations.
                
                Remember:
                
                - preferences
                - stable personal facts
                - goals
                - ongoing projects
                - important decisions
                
                Do NOT remember:
                
                - greetings
                - casual conversation
                - temporary information
                - questions
                - information unlikely to be useful later
                
                Also assign an importance score.
                
                Importance:
                
                0.9 - 1.0 = highly important and stable
                0.6 - 0.8 = useful long-term information
                0.3 - 0.5 = somewhat useful
                0.0 - 0.2 = temporary or low-value
                
                Return ONLY valid JSON.
                
                If it should be remembered:
                
                {{
                    "should_remember": true,
                    "category": "preference",
                    "memory": "User prefers Python for development.",
                    "importance": 0.9
                }}
                
                If it should not:
                
                {{
                    "should_remember": false,
                    "category": null,
                    "memory": null,
                    "importance": 0.0
                }}
                
                User message:
                
                {user_message}
                """

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        memory_decision = json.loads(
            response.output_text
        )

        if not memory_decision[
            "should_remember"
        ]:

            return {
                **memory_decision,
                "action": "ignore",
                "memory_index": None
            }

        new_memory = (
            memory_decision["memory"]
        )

        category = (
            memory_decision["category"]
        )

        if existing_memories:

            memories_text = "\n".join(
                f"{i}: "
                f"[{item['category']}] "
                f"{item['content']}"
                for i, item in enumerate(
                    existing_memories
                )
            )

        else:

            memories_text = (
                "No existing memories."
            )

        update_prompt = f"""
                        You manage long-term memory.
                        Existing memories:
                        
                        {memories_text}
                        
                        New information:
                        
                        [{category}] {new_memory}
                        
                        Determine what should happen.
                        
                        Choose exactly one:
                        
                        "add"
                        - New information.
                        
                        "update"
                        - New information makes an existing
                          memory obsolete or contradictory.
                        
                        "refine"
                        - New information adds useful context
                          to an existing memory.
                        
                        "ignore"
                        - No change needed.
                        
                        If information contradicts an existing
                        memory, use "update".
                        
                        Return ONLY valid JSON.
                        
                        ADD:
                        
                        {{
                            "action": "add",
                            "memory_index": null,
                            "memory": null
                        }}
                        
                        UPDATE:
                        
                        {{
                            "action": "update",
                            "memory_index": 0,
                            "memory": "Updated memory"
                        }}
                        
                        REFINE:
                        
                        {{
                            "action": "refine",
                            "memory_index": 0,
                            "memory": "More precise memory"
                        }}
                        
                        IGNORE:
                        
                        {{
                            "action": "ignore",
                            "memory_index": null,
                            "memory": null
                        }}
                        """

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=update_prompt
        )

        update_decision = json.loads(
            response.output_text
        )

        return {
            **memory_decision,
            **update_decision
        }

    # -------------------------
    # Consolidation
    # -------------------------

    def consolidate_memories(
        self,
        memories_text
    ):

        prompt = f"""
                You are a memory consolidation system.
                
                Stored memories:
                
                {memories_text}
                
                Create a smaller set of durable,
                useful memories.
                
                Rules:
                
                - Remove redundant memories.
                - Combine related memories.
                - Preserve important facts.
                - Preserve current preferences.
                - Preserve current information when facts changed.
                - Do not invent information.
                - Do not combine unrelated facts.
                - Prefer concise statements.
                - Keep useful information for future conversations.
                
                Return ONLY valid JSON:
                
                {{
                    "memories": [
                        {{
                            "category": "preference",
                            "content": "User prefers Python for AI projects.",
                            "importance": 0.9
                        }}
                    ]
                }}
                """

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        result = json.loads(
            response.output_text
        )

        return result.get(
            "memories",
            []
        )

    # -------------------------
    # Benchmark evaluation
    # -------------------------

    def evaluate_answer(
        self,
        question,
        expected,
        actual
    ):

        prompt = f"""
                Evaluate whether the assistant's answer
                correctly answers the question.
                
                Question:
                {question}
                
                Expected answer:
                {expected}
                
                Actual answer:
                {actual}
                
                The answer is correct only if it actually
                provides the expected information.
                
                Do not give credit merely because the
                expected word appears somewhere in the
                response.
                
                Return ONLY:
                
                {{
                    "correct": true
                }}
                
                or:
                
                {{
                    "correct": false
                }}
                """

        response = self.client.responses.create(
            model="gpt-5-mini",
            input=prompt
        )

        result = json.loads(
            response.output_text
        )

        return result["correct"]