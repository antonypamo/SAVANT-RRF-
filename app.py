# ============================================================
# 🚀 SAVANT-RRF- : Chat AGI + Dataset Maestro RRF (Colab/Gradio)
# ============================================================

import gradio as gr
from transformers import pipeline
import json
import os

# -------------------------------
# 1) Cargar dataset de ecuaciones
# -------------------------------
dataset_path = os.path.join(os.path.dirname(__file__), "dataset_rrf.json")
if os.path.exists(dataset_path):
    with open(dataset_path, "r") as f:
        ecuaciones = json.load(f)
else:
    ecuaciones = {"ecuaciones_maestras": []}

# -------------------------------
# 2) Modelos soportados
# -------------------------------
MODELOS = {
    "Ligero (distilgpt2)": "distilgpt2",
    "Avanzado (Falcon-7B-Instruct)": "tiiuae/falcon-7b-instruct",
    "Avanzado (Mistral-7B-Instruct)": "mistralai/Mistral-7B-Instruct-v0.2"
}

modelo_activo = MODELOS["Ligero (distilgpt2)"]
chatbot = pipeline("text-generation", model=modelo_activo)

# -------------------------------
# 3) Funciones auxiliares
# -------------------------------
def cambiar_modelo(nombre):
    global chatbot, modelo_activo
    modelo_activo = MODELOS[nombre]
    chatbot = pipeline("text-generation", model=modelo_activo)
    return f"✅ Modelo cambiado a: {nombre}"

def responder(mensaje, historial):
    # Si el mensaje coincide con una ecuación RRF
    for eq in ecuaciones.get("ecuaciones_maestras", []):
        if eq["id"] in mensaje or eq["titulo"].lower() in mensaje.lower():
            respuesta = f"📘 **{eq['titulo']}**\n\n{eq['ecuacion']}\n\n📖 {eq['descripcion']}"
            historial = historial + [(mensaje, respuesta)]
            return historial, historial

    # Caso contrario, usa el modelo de lenguaje
    respuesta = chatbot(mensaje, max_length=150, num_return_sequences=1, do_sample=True)[0]["generated_text"]
    historial = historial + [(mensaje, respuesta)]
    return historial, historial

# -------------------------------
# 4) Interfaz Gradio
# -------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SAVANT-RRF- Chat AGI + Dataset Maestro RRF")

    with gr.Row():
        modelo_selector = gr.Dropdown(list(MODELOS.keys()), value="Ligero (distilgpt2)", label="Selecciona Modelo")
        salida_modelo = gr.Textbox(label="Estado del modelo")

    modelo_selector.change(cambiar_modelo, modelo_selector, salida_modelo)

    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Escribe aquí tu mensaje (ej: 'muéstrame el Hamiltoniano')")
    clear = gr.Button("🧹 Limpiar Chat")

    msg.submit(responder, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: [], None, chatbot_ui, queue=False)

demo.launch(share=True)
