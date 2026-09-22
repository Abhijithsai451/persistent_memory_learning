import json
import os
from datetime import datetime


class MemoryManager:

    def __init__(self, memory_file="memory.json"):
        self.memory_file = memory_file
        self.data = self.load_memory()

        def load_memory(self):
            if not os.path.exists(self.memory_file):
                return []

            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)

        def save_to_file(self):
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)

        def reset(self):
            self.data = []
            self.save_to_file()

        def get_memories(self):
            return self.data.copy()

        def get_memory_count(self):
            return len(self.data)

        def add_memory(self, category, content, importance=0.5):

            self.data.append({
                "category": category,
                "content": content,
                "importance": importance,
                "created_at": datetime.now().isoformat()
            })

            self.save_to_file()

            print("Memory added successfully.")

        def update_memory(
                self,
                index,
                category,
                content,
                importance=0.5
        ):

            self.data[index] = {
                "category": category,
                "content": content,
                "importance": importance,
                "updated_at": datetime.now().isoformat()
            }

            self.save_to_file()

            print("Memory updated successfully.")

        def search_memory(self, query, limit=5):

            if not self.data:
                return []

            query_words = set(
                query.lower().split()
            )

            scored = []

            for memory in self.data:

                content_words = set(
                    memory["content"].lower().split()
                )

                overlap = len(
                    query_words.intersection(content_words)
                )

                if overlap > 0:
                    importance = memory.get(
                        "importance",
                        0.5
                    )

                    score = overlap + importance

                    scored.append(
                        (score, memory)
                    )

            scored.sort(
                key=lambda item: item[0],
                reverse=True
            )

            return [
                memory
                for _, memory in scored[:limit]
            ]

        def save_memory(self, decision):

            if not decision["should_remember"]:
                print("Memory not saved.")
                return

            action = decision["action"]

            importance = decision.get(
                "importance",
                0.5
            )

            if action == "add":

                self.add_memory(
                    decision["category"],
                    decision["memory"],
                    importance
                )

            elif action == "update":

                self.update_memory(
                    decision["memory_index"],
                    decision["category"],
                    decision["memory"],
                    importance
                )

            elif action == "refine":

                self.update_memory(
                    decision["memory_index"],
                    decision["category"],
                    decision["memory"],
                    importance
                )

            elif action == "ignore":

                print("Memory ignored.")

        def should_consolidate(self, threshold=5):
            return len(self.data) >= threshold

        def consolidate(self, llm):

            if len(self.data) < 2:
                return

            memories_text = "\n".join(
                f"{i}: "
                f"[{memory['category']}] "
                f"(importance={memory.get('importance', 0.5)}) "
                f"{memory['content']}"
                for i, memory in enumerate(self.data)
            )

            consolidated = llm.consolidate_memories(
                memories_text
            )

            if not consolidated:
                return

            self.data = consolidated

            self.save_to_file()

            print("Memory consolidation completed.")