
import gradio as gr
from transformers import pipeline

# Modelo base HF (ligero para PC de gama baja / Colab)
chatbot = pipeline("text-generation", model="distilgpt2")

def responder(mensaje, historial):
    respuesta = chatbot(mensaje, max_length=150, num_return_sequences=1, do_sample=True)[0]["generated_text"]
    return respuesta, historial + [(mensaje, respuesta)]

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SAVANT-RRF- Chat AGI Experimental")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Escribe aquí tu mensaje")
    clear = gr.Button("🧹 Limpiar Chat")

    msg.submit(responder, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: None, None, chatbot_ui, queue=False)

demo.launch(share=True)  # 👈 share=True para que abra URL pública en Colab
