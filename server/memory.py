from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Optional
from datetime import datetime
import json
import os


class MemoryStore:
    def __init__(self):
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.memories: List[Dict] = []
        self.vector_dimension = (
            384  # Vector dimension for paraphrase-multilingual-MiniLM-L12-v2
        )
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        self._load_memories()

    def _save_memories(self):
        """Save memory content to files"""
        memory_file = "memories_meta.json"
        with open(memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

        # Save FAISS index
        index_file = "memories.index"
        faiss.write_index(self.index, index_file)

    def _load_memories(self):
        """Load memory content and vector index from files"""
        memory_file = "memories_meta.json"
        index_file = "memories.index"

        # Load memory metadata
        if os.path.exists(memory_file):
            with open(memory_file, "r", encoding="utf-8") as f:
                self.memories = json.load(f)

        # Load or initialize FAISS index
        if os.path.exists(index_file):
            self.index = faiss.read_index(index_file)
        else:
            # If we have memory metadata but no index file, rebuild the index
            if self.memories:
                vectors = [self.model.encode(m["content"]) for m in self.memories]
                vectors_array = np.array(vectors).astype("float32")
                self.index.add(vectors_array)

    def add_memory(self, content: str, tags: Optional[List[str]] = None) -> Dict:
        """Add a new memory entry

        Args:
            content: The content of the memory
            tags: Optional list of tags for the memory

        Returns:
            Dict containing the newly added memory
        """
        memory = {
            "id": len(self.memories),
            "content": content,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
        }
        # Add memory to list
        self.memories.append(memory)

        # Add vector to FAISS index
        vector = self.model.encode(content)
        self.index.add(np.array([vector]).astype("float32"))

        self._save_memories()
        return memory

    def search_by_keywords(self, keywords: str, top_k: int = 5) -> List[Dict]:
        """Search memories by keywords

        Args:
            keywords: The keywords to search for
            top_k: Maximum number of results to return

        Returns:
            List of matching memories sorted by creation date
        """
        matched_memories = []
        for memory in self.memories:
            # Check if content or tags contain the keywords
            if keywords.lower() in memory["content"].lower() or any(
                keywords.lower() in tag.lower() for tag in memory["tags"]
            ):
                matched_memories.append(memory)

        # Sort by creation date and return top_k results
        return sorted(matched_memories, key=lambda x: x["created_at"], reverse=True)[
            :top_k
        ]

    def search_by_similarity(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search memories by vector similarity

        Args:
            query: The query text to search for
            top_k: Maximum number of results to return

        Returns:
            List of similar memories with similarity scores
        """
        if not self.memories:
            return []

        # Encode query text
        query_vector = self.model.encode(query)
        query_vector = np.array([query_vector]).astype("float32")

        # Use FAISS to search for nearest neighbors
        distances, indices = self.index.search(query_vector, top_k)

        # Build results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.memories):  # Ensure index is valid
                memory = dict(self.memories[idx])
                memory["similarity"] = float(
                    1.0 / (1.0 + distance)
                )  # Convert distance to similarity score
                results.append(memory)

        return results

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory entry by its ID

        Args:
            memory_id: The ID of the memory to delete

        Returns:
            bool: True if memory was successfully deleted, False if not found
        """
        # Find the memory index in the list
        try:
            memory_index = next(
                i for i, m in enumerate(self.memories) if m["id"] == memory_id
            )
        except StopIteration:
            return False

        # Remove the memory from the list
        del self.memories[memory_index]

        # Rebuild FAISS index
        self.index = faiss.IndexFlatL2(self.vector_dimension)
        if self.memories:
            vectors = [self.model.encode(m["content"]) for m in self.memories]
            vectors_array = np.array(vectors).astype("float32")
            self.index.add(vectors_array)

        self._save_memories()
        return True


mcp = FastMCP("memory", host="127.0.0.1", port=8001)
memory_store = MemoryStore()


@mcp.tool()
def search_memories(keywords: str, max_results: int = 5) -> List[Dict]:
    """
    Search memories by keywords.

    Args:
        keywords: Keywords to search for in memory content and tags
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of matching memories sorted by creation date
    """
    return memory_store.search_by_keywords(keywords, max_results)


@mcp.tool()
def add_memory(content: str, tags: Optional[List[str]] = None) -> Dict:
    """
    Add a new memory entry.

    Args:
        content: Content of the memory
        tags: Optional list of tags for categorizing the memory

    Returns:
        Dictionary containing the newly added memory
    """
    return memory_store.add_memory(content, tags)


@mcp.tool()
def search_similar_memories(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search memories by semantic similarity using vector embeddings.

    Args:
        query: Query text to find similar memories
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of similar memories sorted by similarity score
    """
    return memory_store.search_by_similarity(query, max_results)


@mcp.tool()
def delete_memory(memory_id: int) -> bool:
    """
    Delete a memory entry by its ID.

    Args:
        memory_id: The ID of the memory entry to delete

    Returns:
        bool: True if the memory was successfully deleted, False if not found
    """
    return memory_store.delete_memory(memory_id)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
