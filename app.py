import streamlit as st
import pickle
import faiss
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from openai import OpenAI
import json


@dataclass
class HelpCenterRAG:
    """Class to handle help center retrieval augmented generation with FAISS."""

    model: SentenceTransformer
    qa_data: List[dict]
    index: faiss.IndexFlatIP = None

    def __post_init__(self):
        """Initialize FAISS index if not provided."""
        if self.index is None:
            # Create text sections for embeddings (excluding source).
            sections = [
                f"Q: {item['question']}\nA: {item['answer']}\n文档位置：{item['document_location']}"
                for item in self.qa_data
            ]
            # Embed the sections.
            embeddings = self.model.encode(sections, convert_to_tensor=False)

            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product index

            # Normalize vectors to convert inner product to cosine similarity
            faiss.normalize_L2(embeddings)

            # Add vectors to the index
            self.index.add(embeddings)

    def find_relevant_sections(
        self, query: str, top_k: int = 10, min_score: float = 0.4
    ) -> List[dict]:
        """
        Find the most relevant sections using FAISS.

        Args:
            query: User's question
            top_k: Number of relevant sections to return
            min_score: Minimum similarity score threshold

        Returns:
            A list of dicts containing relevant QA data plus similarity scores.
        """
        # Get query embedding and normalize it
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        # Filter results by min_score and format output
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_score:
                results.append(
                    {
                        "question": self.qa_data[idx]["question"],
                        "answer": self.qa_data[idx]["answer"],
                        "document_location": self.qa_data[idx]["document_location"],
                        "source": self.qa_data[idx]["source"],
                        # "similarity_score": float(score)  # If needed
                    }
                )
        return results

    def save(self, filepath: str):
        """Save the RAG instance to a file."""
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "qa_data": self.qa_data,
                    "index_data": faiss.serialize_index(self.index),
                },
                f,
            )

    @classmethod
    def load(cls, filepath: str, model: SentenceTransformer):
        """Load a RAG instance from a file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        index = faiss.deserialize_index(data["index_data"])
        return cls(model=model, qa_data=data["qa_data"], index=index)


def get_deepseek_completion(query: str, context: List[Dict]) -> Dict:
    """
    Make a request to DeepSeek/OpenAI API with the provided query and context.
    Returns the LLM response and token usage.
    """
    # Replace with your actual API key and base_url
    client = OpenAI(
        api_key="sk-d76d8416a0f8433d8a3545673af2d75f",
        base_url="https://api.deepseek.com",
    )

    # Format the context for prompt
    context_json = json.dumps(context, ensure_ascii=False, indent=2)

    # Construct conversation
    messages = [
        {
            "role": "system",
            "content": (
                "FastBull网站是一家全球知名的金融服务提供商。"
                "我将提供部分FastBull网站帮助中心的回答文档，请你根据提供的文档回答用户问题。\n\n"
                "以下是回答问题时的一些要求：\n"
                "- 请根据提供的信息，对用户问题给出简明直接的答案。语气自然专业，时刻牢记你服务的平台是FastBull，并将用户放在首位。\n"
                "- 如果提供的信息无法直接回答问题，请说：帮助中心可能没有针对这个问题的直接答案，建议（其他选项如联系客服或浏览相关分类）。\n"
                "- 如果用户问的问题过于宽泛，可能提供的帮助中心文档会不全面，这时请指出该问题并建议用户提出更具体的问题或自行查看帮助中心文档。\n"
                "- 回答问题时无需说明参考了帮助中心文档。\n"
                "- 如果可能，请注明信息来源的网址。\n"
                "- 你可以拒绝回答闲聊或无关紧要的问题。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"以下是帮助中心文档中可能有助于回答用户问题的相关问答对：\n{context_json}"
                f'\n用户问题："""{query}"""'
            ),
        },
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
    )

    # Retrieve text and token usage from the response
    result = response.choices[0].message.content
    usage = {
        "prompt_cache_hit_tokens": response.usage.prompt_cache_hit_tokens,
        "prompt_cache_miss_tokens": response.usage.prompt_cache_miss_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return {"result": result, "usage": usage}


def main():
    st.title("FastBull Help Center RAG Demo")

    # --- Load model and RAG on first run or in cache ---
    if "rag" not in st.session_state:
        st.write("Loading model and RAG index (this might take a moment)...")
        # Replace with your own model
        model = SentenceTransformer("TencentBAC/Conan-embedding-v1")
        # Replace with your actual file path
        rag = HelpCenterRAG.load("help_center_rag_faiss.pkl", model)
        st.session_state["rag"] = rag
        st.success("RAG loaded successfully!")
    else:
        rag = st.session_state["rag"]

    # User input
    query = st.text_input(
        "请输入您的问题 (Enter your query):", value="我要怎么在平台上赚钱？"
    )

    # Submit query
    if st.button("提交问题 (Submit)"):
        with st.spinner("正在查询帮助中心..."):
            context = rag.find_relevant_sections(query)
        # st.write("以下是检索到的相关信息：", context)

        with st.spinner("正在向DeepSeek API发送请求..."):
            response_data = get_deepseek_completion(query, context)
            result = response_data["result"]
            usage = response_data["usage"]

        # Display the result
        # st.markdown("### 回答 (Response):")
        st.markdown(result)

        # Calculate approximate request cost
        # Adjust the cost formula to match your actual pricing/usage model
        cost = (
            usage["prompt_cache_hit_tokens"] * 0.1 / 1_000_000
            + usage["prompt_cache_miss_tokens"] * 1 / 1_000_000
            + usage["completion_tokens"] * 2 / 1_000_000
        )
        st.write(f"请求成本 (Approx. Cost): {cost:.6f} 元")


if __name__ == "__main__":
    main()
