import os
import requests
import time
import threading
import runpod
import json
from websocket_server import WebsocketServer
from llama_cpp import Llama

MAX_SESSION_SECONDS = int(os.environ.get("MAX_SESSION_SECONDS", "30"))
MODEL_URL = os.environ.get("MODEL_URL")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
WS_PORT = int(os.environ.get("WS_PORT", "443"))
N_CTX = int(os.environ.get("N_CTX", "16384"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "999"))  # try to offload as much as possible

llm = None
shutdown_flag = False

def download_model():
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var is required")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return

    with requests.get(MODEL_URL, stream=True) as r:
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def init_model():
    global llm
    if llm is None:
        download_model()
        llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            verbose=False,
        )

def on_message(client, server, message: str):
    global shutdown_flag
    msg = (message or "").strip()

    if msg.lower() == "shutdown":
        shutdown_flag = True
        server.send_message(client, json.dumps({"type": "shutdown"}))
        server.shutdown()
        return

    # Parse OpenAI-like payload
    try:
        payload = json.loads(msg)
    except json.JSONDecodeError:
        server.send_message(client, json.dumps({"type": "error", "message": "Invalid JSON"}))
        return

    messages = payload.get("messages", [])
    temperature = float(payload.get("temperature", 0.7))
    max_tokens = int(payload.get("max_tokens", 512))

    # Minimal chat prompt (you can improve this later)
    prompt = ""
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = m.get("content") or ""
        prompt += f"{role}: {content}\n"
    prompt += "ASSISTANT: "

    init_model()

    # Stream token deltas
    try:
        for out in llm(prompt=prompt, max_tokens=max_tokens, stream=True, temperature=temperature):
            token = out["choices"][0]["text"]
            if token:
                server.send_message(client, json.dumps({"type": "delta", "content": token}))
        server.send_message(client, json.dumps({"type": "done"}))
    except Exception as e:
        server.send_message(client, json.dumps({"type": "error", "message": str(e)}))


def start_websocket():
    global shutdown_flag
    shutdown_flag = False

    server = WebsocketServer(host="0.0.0.0", port=WS_PORT)
    server.set_fn_message_received(on_message)

    # Safety timer: shuts down even if client never sends "shutdown"
    def auto_shutdown():
        time.sleep(MAX_SESSION_SECONDS)
        if not shutdown_flag:
            server.shutdown()

    threading.Thread(target=auto_shutdown, daemon=True).start()

    server.run_forever()
    return "WebSocket server stopped"

def handler(event):
    public_ip = os.environ.get("RUNPOD_PUBLIC_IP", "localhost")
    tcp_port = int(os.environ.get(f"RUNPOD_TCP_PORT_{WS_PORT}", str(WS_PORT)))

    print(f"CONNECT ws://{public_ip}:{tcp_port}", flush=True)

    init_model()
    result = start_websocket()

    return {"message": result, "public_ip": public_ip, "tcp_port": tcp_port}

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
