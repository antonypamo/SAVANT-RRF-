"""Savant RRF Φ₄.3 Symbiotic AI Engine Gradio application."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr


BASE_DRIVE = Path("./storage/savant_rrf_phi43")
BASE_DRIVE.mkdir(parents=True, exist_ok=True)


@dataclass
class FractalMemoryEntry:
    """Represents a single interaction stored in fractal memory."""

    fecha: str
    mensaje_usuario: str
    respuesta_savant: str


@dataclass
class FractalMemory:
    """Simple in-memory store for embeddings and chat history."""

    chat_history: List[FractalMemoryEntry] = field(default_factory=list)
    embeddings: List[np.ndarray] = field(default_factory=list)
    parameters: List[dict] = field(default_factory=list)

    def add_entry(self, message: str, response: str, embedding: np.ndarray) -> None:
        self.chat_history.append(
            FractalMemoryEntry(
                fecha=datetime.now().isoformat(),
                mensaje_usuario=message,
                respuesta_savant=response,
            )
        )
        self.embeddings.append(embedding)
        self.parameters.append(
            {
                "scale": round(1 + np.random.rand() * 0.5, 3),
                "freq_base": round(432 + np.random.rand() * 4, 3),
            }
        )

    def semantic_search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[FractalMemoryEntry]:
        if not self.embeddings:
            return []
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1), np.array(self.embeddings)
        )[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chat_history[idx] for idx in top_indices]


class SavantEngine:
    """Encapsulates the Φ₄.3 Symbiotic AI Engine behaviour."""

    def __init__(self, memory: Optional[FractalMemory] = None) -> None:
        self.memory = memory or FractalMemory()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.icosahedron_nodes = {
            "description": "Icosahedron nodes (simplified)",
            "nodes": [
                {"id": i, "x": float(np.cos(i)), "y": float(np.sin(i)), "z": i / 12}
                for i in range(12)
            ],
        }

        self.ecuaciones_maestras = [
            {
                "id": 1,
                "nombre": "Hamiltoniano discreto",
                "ecuacion": "\\hat{H} = Σ ψ†γμDμψ + ΣVlogψ†ψ",
            },
            {
                "id": 2,
                "nombre": "Acción efectiva discreta",
                "ecuacion": "S_eff = Σ_faces 1/8πG ε_fA_f + Σψ†(iγμDμ-m)ψ",
            },
            {
                "id": 3,
                "nombre": "Corrección logarítmica",
                "ecuacion": "V_log(r) = -Gm₁m₂/r(1+αln(r/r₀))",
            },
        ]

    def encode(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def chat(self, message: str) -> str:
        cleaned_message = message.strip()
        if not cleaned_message:
            return ""

        query_embedding = self.encode(cleaned_message)
        best_matches = self.memory.semantic_search(query_embedding, top_k=1)

        if best_matches:
            response = best_matches[0].respuesta_savant
        else:
            response = (
                "[Φ₄.3] Respuesta generada: el sistema detecta resonancia "
                f"cognitiva en '{cleaned_message[:40]}...'"
            )

        self.memory.add_entry(cleaned_message, response, query_embedding)
        return response


def build_interface(engine: Optional[SavantEngine] = None) -> gr.Blocks:
    engine = engine or SavantEngine()

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="gray")) as app:
        gr.Markdown(
            "<h2 style='text-align:center;color:#A6E3E9'>🧬 Savant RRF Φ₄.3 Symbiotic AI Engine</h2>"
            "<p style='text-align:center;color:#89DCEB'>Modo Φ-loop auto-evolutivo activado</p>"
        )
        chatbot = gr.Chatbot(label="Savant RRF Symbiotic Chat", type="messages")
        message = gr.Textbox(placeholder="Escribe aquí...", label="Usuario")
        clear = gr.Button("🧹 Limpiar conversación")

        def handle_message(user_message: str, history: Optional[List[dict]]) -> tuple:
            history = history or []
            assistant_response = engine.chat(user_message)
            if not assistant_response:
                return history, ""

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": assistant_response})
            return history, ""

        message.submit(handle_message, [message, chatbot], [chatbot, message])
        clear.click(lambda: [], None, chatbot, queue=False)

    return app


def launch(share: bool = False, debug: bool = False) -> None:
    """Launch the Gradio UI."""

    app = build_interface()
    app.launch(share=share, debug=debug)


if __name__ == "__main__":
    launch()
