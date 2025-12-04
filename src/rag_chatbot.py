# src/rag_chatbot.py

from typing import List, Tuple
import textwrap

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from .rag_index import LivermoreIndex


class LivermoreRAGChatbot:
    """
    RAG chatbot that:
    - retrieves relevant Q&A entries about Jesse Livermore
    - uses FLAN-T5 to generate an answer in his voice
    """

    def __init__(
        self,
        excel_path: str = "data/Team Livermore.xlsx",
        flan_model_name: str = "google/flan-t5-base",
        device: str = "cpu",
    ):
        # 1. Retrieval components
        self.index = LivermoreIndex(excel_path)

        # 2. Generation components
        self.tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)
        self.device = device
        self.model.to(self.device)

    def _build_context(self, retrieved: List[dict]) -> str:
        """
        Build a textual context from retrieved Q&A entries.
        """
        blocks = []
        for r in retrieved:
            block = (
                f"Q: {r['question']}\n"
                f"A: {r['answer']}\n"
                f"(Label: {r['label']})"
            )
            blocks.append(block)
        context = "\n\n".join(blocks)
        return context

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Construct the instruction prompt for FLAN-T5.
        """
        prompt = f"""
        You are Jesse Livermore, a legendary stock trader. Answer the question in first person ("I")
        using only the information from the context below. Stay faithful to my real philosophy and style.
        If the context is insufficient, say that I haven't directly addressed this and answer cautiously
        based only on what is given.

        Context:
        {context}

        Question: {question}

        Answer as Jesse Livermore in 3–5 sentences. Explain your reasoning clearly.
        """
        prompt = textwrap.dedent(prompt).strip()
        return prompt

    def answer_question(
        self,
        question: str,
        k: int = 3,
        max_new_tokens: int = 300,
    ) -> Tuple[str, List[dict]]:
        """
        Retrieve top-k relevant QAs and generate an answer as Jesse Livermore.

        Returns:
          answer_text: str
          sources: List[dict] -> [{'question', 'answer', 'label', 'score'}, ...]
        """
        # 1. Retrieve context
        retrieved = self.index.retrieve(question, k=k)

        if not retrieved:
            answer = (
                "I don't have any direct teachings in my records about this exact question. "
                "You may want to rephrase or ask something more specific about my trading or philosophy."
            )
            return answer, []

        # 2. Build context + prompt
        context = self._build_context(retrieved)
        prompt = self._build_prompt(question, context)

        # 3. Tokenize & generate
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic (greedy)
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.strip()

        return answer, retrieved


if __name__ == "__main__":
    # Quick interactive test: python -m src.rag_chatbot
    bot = LivermoreRAGChatbot("data/Team Livermore.xlsx")

    print("Chat with Jesse Livermore (type 'exit' to quit)\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            break

        ans, _sources = bot.answer_question(user_q, k=3)
        print("Jesse:", ans)
        print("-" * 80)