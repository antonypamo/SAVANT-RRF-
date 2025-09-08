
import gradio as gr
from transformers import pipeline

# Modelos soportados
MODELOS = {
    "Ligero (distilgpt2)": "distilgpt2",
    "Avanzado (Falcon-7B-Instruct)": "tiiuae/falcon-7b-instruct",
    "Avanzado (Mistral-7B-Instruct)": "mistralai/Mistral-7B-Instruct-v0.2"
}

# Inicializar con modelo ligero por defecto
modelo_activo = MODELOS["Ligero (distilgpt2)"]
chatbot = pipeline("text-generation", model=modelo_activo)

def cambiar_modelo(nombre):
    global chatbot, modelo_activo
    modelo_activo = MODELOS[nombre]
    chatbot = pipeline("text-generation", model=modelo_activo)
    return f"✅ Modelo cambiado a: {nombre}"

def responder(mensaje, historial):
    respuesta = chatbot(mensaje, max_length=150, num_return_sequences=1, do_sample=True)[0]["generated_text"]
    historial = historial + [(mensaje, respuesta)]
    return historial, historial

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 SAVANT-RRF- Chat AGI Experimental (con selector de modelo)")

    with gr.Row():
        modelo_selector = gr.Dropdown(list(MODELOS.keys()), value="Ligero (distilgpt2)", label="Selecciona Modelo")
        salida_modelo = gr.Textbox(label="Estado del modelo")

    modelo_selector.change(cambiar_modelo, modelo_selector, salida_modelo)

    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Escribe aquí tu mensaje")
    clear = gr.Button("🧹 Limpiar Chat")

    msg.submit(responder, [msg, chatbot_ui], [chatbot_ui, chatbot_ui])
    clear.click(lambda: [], None, chatbot_ui, queue=False)

demo.launch(share=True)  # Esto abre el link público de Gradio/Hugging Face
