import gradio as gr
from fractal_memory import FractalMemory
from rrf_ai_model import RRF_AI_Model

# Inicializa la memoria y el modelo
fractal_memory_instance = FractalMemory()
rrf_config = {"phi": (1 + 5**0.5) / 2, "lambda_corr": 0.1}
rrf_model = RRF_AI_Model(fractal_memory_instance, rrf_config)

def chat_ai(user_input, history):
    response = rrf_model.process_input_with_memory(user_input)
    history = history or []
    history.append((user_input, response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# Savant-RRF AI\nConversación demo con memoria fractal y RRF")
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Escribe tu mensaje aquí...",
            container=False
        )
    state = gr.State([])
    send = gr.Button("Enviar")
    send.click(chat_ai, inputs=[txt, state], outputs=[chatbot, state])
    txt.submit(chat_ai, inputs=[txt, state], outputs=[chatbot, state])

if __name__ == "__main__":
    demo.launch()