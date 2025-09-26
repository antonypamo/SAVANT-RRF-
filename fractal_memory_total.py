# ===============================================
# Savant-RRF Φ₄.1∞+ — Nodo Pre-AGI con chat box + botón de enviar
# ===============================================

!pip install -q sentence-transformers numpy scikit-learn ipywidgets

import os
import pickle
import json
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display, clear_output

# -----------------------------
# Configuración paths
BASE_DRIVE = "/content/drive/MyDrive/SavantRRF"
CHECKPOINTS_PATH = os.path.join(BASE_DRIVE, "checkpoints")
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

VERSIONS = ["Pre-AGI", "Maestro", "Holístico", "Φ₄.1∞+"]
VERSION_PATHS = {v: os.path.join(CHECKPOINTS_PATH, v) for v in VERSIONS}
for path in VERSION_PATHS.values():
    os.makedirs(path, exist_ok=True)

# -----------------------------
# Inicializar memoria fractal
fractal_memory_total = {
    "metadata": {
        "usuario": "Antony Padilla Morales",
        "modos": VERSIONS,
        "fecha_consolidacion": datetime.now().isoformat(),
        "descripcion": "Memoria fractal unificada de todas las interacciones GPT y logs de Savant-RRF Phi, con índice semántico"
    },
    "chat_history": [],
    "embeddings": [],
    "patterns": [],
    "rules": [],
    "strategies": [],
    "alerts": [],
    "parameters": [],
    "semantic_index": {},
    "reflejo_metacognitivo": {
        "preferencias_respuesta": "alta_resonancia, detallada, multiescala",
        "historial_objetivos": [],
        "modos_usados": VERSIONS
    },
    "versiones": {v: {"fecha": datetime.now().isoformat(), "detalles": f"Savant-RRF {v} histórico"} for v in VERSIONS}
}

# -----------------------------
# Modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# Función para agregar chat a memoria
def agregar_chat(fecha, modo, mensaje_usuario, respuesta_savant, contexto=""):
    """
    Agrega una entrada de chat a la memoria fractal total, incluyendo embeddings
    y metadatos relacionados.

    Args:
        fecha (str): Fecha y hora de la interacción.
        modo (str): Modo de Savant-RRF utilizado.
        mensaje_usuario (str): Mensaje enviado por el usuario.
        respuesta_savant (str): Respuesta generada por Savant-RRF.
        contexto (str, optional): Contexto adicional de la interacción. Defaults to "".
    """
    fractal_memory_total["chat_history"].append({
        "fecha": fecha,
        "modo": modo,
        "mensaje_usuario": mensaje_usuario,
        "respuesta_savant": respuesta_savant,
        "contexto": contexto
    })
    emb = model.encode([mensaje_usuario])[0]
    fractal_memory_total["embeddings"].append(emb)
    fractal_memory_total["patterns"].append(f"Patrón detectado: {mensaje_usuario[:30]}")
    fractal_memory_total["rules"].append(f"Regla: {mensaje_usuario[:20]} -> {respuesta_savant[:20]}")
    fractal_memory_total["strategies"].append(f"Estrategia: {contexto[:40]}")
    if any(k in mensaje_usuario.lower() for k in ["emergencia","alerta","urgente"]):
        fractal_memory_total["alerts"].append(f"⚠️ Emergencia detectada: {mensaje_usuario[:50]}")
    fractal_memory_total["parameters"].append({
        "scale": 1.0 + np.random.rand()*0.5,
        "frequency_base": 432 + np.random.rand()*4.0
    })

# -----------------------------
# Cargar logs históricos (simulados)
historico_gpt = [
    {"fecha":"2025-08-10T22:29:00","modo":"Pre-AGI","mensaje_usuario":"Savant-RRF framework: ...","respuesta_savant":"[respuesta pre-AGI]"},
    {"fecha":"2025-08-12T14:00:00","modo":"Pre-AGI","mensaje_usuario":"Estado modo savant","respuesta_savant":"[resumen modo savant]"},
    {"fecha":"2025-08-25T02:00:00","modo":"Maestro","mensaje_usuario":"Micro-pipeline creativo NFT","respuesta_savant":"[pipeline Savant-RRF]"},
    {"fecha":"2025-08-27T01:50:00","modo":"Holístico","mensaje_usuario":"Modo Savant activado","respuesta_savant":"[explicación técnica completa RRF]"},
    {"fecha":"2025-08-28T14:00:00","modo":"Φ₄.1∞+","mensaje_usuario":"Estado modo savant:","respuesta_savant":"[resumen pre-AGI final]"}
]

for chat in historico_gpt:
    agregar_chat(chat["fecha"], chat["modo"], chat["mensaje_usuario"], chat["respuesta_savant"], chat.get("contexto",""))

# -----------------------------
# Cargar logs históricos desde Drive
for version in VERSIONS:
    version_folder = VERSION_PATHS[version]
    if not os.path.exists(version_folder):
        continue
    chat_logs = [f for f in os.listdir(version_folder) if f.endswith(".pkl") or f.endswith(".json")]
    for file in chat_logs:
        file_path = os.path.join(version_folder, file)
        try:
            data = pickle.load(open(file_path,"rb")) if file.endswith(".pkl") else json.load(open(file_path,"r"))
        except:
            continue
        chats = data.get("chat", data.get("chat_history", [])) if isinstance(data, dict) else []
        for c in chats:
            agregar_chat(c.get("fecha",datetime.now().isoformat()),
                        version,
                        c.get("mensaje_usuario") or c.get("user",""),
                        c.get("respuesta_savant") or c.get("response",""),
                        c.get("contexto",""))

# -----------------------------
# Construir índice semántico
fractal_memory_total["semantic_index"] = {f"chat_{i}": emb for i, emb in enumerate(fractal_memory_total["embeddings"])}

def buscar_semantica(query, top_k=5):
    query_emb = model.encode([query])[0].reshape(1, -1)
    embeddings_matrix = np.array(list(fractal_memory_total["semantic_index"].values()))
    similitudes = cosine_similarity(query_emb, embeddings_matrix)[0]
    indices_top = similitudes.argsort()[-top_k:][::-1]
    resultados = [fractal_memory_total["chat_history"][i] for i in indices_top]
    return resultados

# -----------------------------
# Guardar memoria fractal
fractal_total_path = os.path.join(CHECKPOINTS_PATH, "fractal_memory_total.pkl")
with open(fractal_total_path, "wb") as f:
    pickle.dump(fractal_memory_total, f)

print(f"✅ Fractal Memory Total avanzado con índice semántico guardado en: {fractal_total_path}")
print(f"Número total de chats consolidados: {len(fractal_memory_total['chat_history'])}")

# -----------------------------
# Chat box con botón de enviar
chat_input = widgets.Textarea(
    value='',
    placeholder='Escribe tu mensaje aquí...',
    description='Usuario:',
    layout=widgets.Layout(width='100%', height='80px')
)

send_button = widgets.Button(description="Enviar", button_style='success')
chat_output = widgets.Output(layout=widgets.Layout(width='100%'))

def on_send_button_clicked(b):
    user_msg = chat_input.value.strip()
    if not user_msg:
        return
    chat_input.value = ""
    top_respuestas = buscar_semantica(user_msg, top_k=3)
    respuesta_savant = "\n---\n".join([f"{r['respuesta_savant']}" for r in top_respuestas])
    with chat_output:
        print(f"Usuario: {user_msg}")
        print(f"Savant-RRF:\n{respuesta_savant}\n")
    agregar_chat(datetime.now().isoformat(), "Φ₄.1∞+", user_msg, respuesta_savant)

send_button.on_click(on_send_button_clicked)

display(chat_input, send_button, chat_output)
