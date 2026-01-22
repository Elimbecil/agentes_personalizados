import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

# =========================
# CONFIG
# =========================
APP_TITLE = "Agent Studio (MVP)"
BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
AGENTS_PATH = STORAGE_DIR / "agents.json"

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
DEFAULT_BUILDER_MODEL = os.getenv("OPENAI_BUILDER_MODEL", "gpt-4.1-mini")

st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="wide")


# =========================
# AUTH (Access Key Gate)
# =========================
def check_access() -> None:
    """
    Bloquea toda la app hasta que el usuario introduzca la clave correcta.
    La clave se define en st.secrets["ACCESS_KEY"] (Streamlit Secrets).
    """
    if st.session_state.get("access_granted"):
        return

    access_key = st.secrets.get("ACCESS_KEY", "")
    if not access_key:
        st.error("Falta ACCESS_KEY en Streamlit Secrets.")
        st.stop()

    st.title("🔒 Acceso restringido")
    st.write("Introduce la clave para acceder.")

    key = st.text_input("Clave de acceso", type="password")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Entrar", use_container_width=True):
            if key == access_key:
                st.session_state["access_granted"] = True
                st.rerun()
            else:
                st.error("Clave incorrecta.")
    with col2:
        st.caption("Si no tienes la clave, contacta con el administrador.")

    st.stop()


# =========================
# OPENAI CLIENT
# =========================
def get_openai_api_key() -> str:
    return st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")


def get_openai_client() -> OpenAI:
    api_key = get_openai_api_key()
    if not api_key:
        st.error("Falta OPENAI_API_KEY. Añádela en Streamlit Secrets (recomendado).")
        st.stop()
    return OpenAI(api_key=api_key)


client = get_openai_client()


# =========================
# STORAGE
# =========================
def ensure_storage() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    if not AGENTS_PATH.exists():
        AGENTS_PATH.write_text("[]", encoding="utf-8")


def load_agents() -> List[Dict[str, Any]]:
    ensure_storage()
    try:
        data = json.loads(AGENTS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_agents(agents: List[Dict[str, Any]]) -> None:
    ensure_storage()
    AGENTS_PATH.write_text(json.dumps(agents, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_agent(agent: Dict[str, Any]) -> None:
    agents = load_agents()
    idx = next((i for i, a in enumerate(agents) if a.get("id") == agent.get("id")), None)
    if idx is None:
        agents.append(agent)
    else:
        agents[idx] = agent
    save_agents(agents)


# =========================
# OPENAI HELPERS
# =========================
def call_responses(model: str, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
    resp = client.responses.create(
        model=model,
        input=messages,
        temperature=temperature,
    )

    text = getattr(resp, "output_text", None)
    if text:
        return text

    # fallback
    try:
        parts = []
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    if c.type == "output_text":
                        parts.append(c.text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    raw = match.group(0).strip()

    try:
        return json.loads(raw)
    except Exception:
        raw2 = raw.strip("` \n")
        try:
            return json.loads(raw2)
        except Exception:
            return None


# =========================
# SESSION ROUTER
# =========================
def goto(page: str) -> None:
    st.session_state["page"] = page


def init_state() -> None:
    st.session_state.setdefault("page", "home")
    st.session_state.setdefault("selected_agent_id", None)

    # Chat histories por agente: {agent_id: [ {role, content}, ... ]}
    st.session_state.setdefault("chat_histories", {})

    # Creator state
    st.session_state.setdefault("builder_messages", [])
    st.session_state.setdefault("agent_draft", {})
    st.session_state.setdefault("agent_ready", False)
    st.session_state.setdefault("built_system_prompt", "")


# =========================
# BUILDER PROMPT
# =========================
BUILDER_SYSTEM = """
Eres "Agent Builder": un asistente que crea agentes con personalidad.
Tu trabajo es hacer preguntas UNA POR UNA para definir un agente conversacional.

Reglas:
- Solo haces 1 pregunta por turno (máximo 2 frases).
- Mantén el flujo simple, en español, tono profesional-amigable.
- Vas acumulando un borrador del agente.
- Al finalizar, marcas done=true y produces un system_prompt final listo para usar.

IMPORTANTE: Debes responder SIEMPRE en JSON, sin texto adicional fuera del JSON.

Formato JSON:
{
  "assistant_message": "pregunta o mensaje breve para el usuario",
  "draft_update": { ... campos que hayas inferido o confirmado ... },
  "done": false,
  "system_prompt": ""  // solo cuando done=true
}

Campos sugeridos para draft_update:
- name
- description
- language (por defecto "es")
- tone
- role
- goals (lista)
- boundaries (lista: lo que NO debe hacer)
- style_rules (lista: cómo responde)
- tools (lista opcional)
- example_user (string opcional)
- example_assistant (string opcional)

Criterio de finalización:
- Cuando tengas al menos: name, description, role, tone, goals (>=2), boundaries (>=2), style_rules (>=3)
- Entonces genera system_prompt consolidado.
""".strip()


def builder_step(user_input: str) -> Dict[str, Any]:
    draft = st.session_state.get("agent_draft", {})
    builder_messages = st.session_state.get("builder_messages", [])

    messages = [
        {"role": "system", "content": BUILDER_SYSTEM},
        {"role": "user", "content": f"BORRADOR ACTUAL (JSON):\n{json.dumps(draft, ensure_ascii=False)}"},
    ]

    messages.extend(builder_messages[-12:])
    messages.append({"role": "user", "content": user_input})

    raw = call_responses(model=DEFAULT_BUILDER_MODEL, messages=messages, temperature=0.4)
    data = extract_json(raw)

    if not data:
        data = {
            "assistant_message": "Se me desordenó el formato. ¿Cómo quieres que se llame el agente y qué hace en una frase?",
            "draft_update": {},
            "done": False,
            "system_prompt": "",
        }

    builder_messages.append({"role": "user", "content": user_input})
    builder_messages.append({"role": "assistant", "content": json.dumps(data, ensure_ascii=False)})
    st.session_state["builder_messages"] = builder_messages

    return data


def merge_draft(update: Dict[str, Any]) -> None:
    draft = st.session_state.get("agent_draft", {})
    for k, v in (update or {}).items():
        if isinstance(v, list) and isinstance(draft.get(k), list):
            seen = set(draft[k])
            draft[k].extend([x for x in v if x not in seen])
        else:
            draft[k] = v
    st.session_state["agent_draft"] = draft


# =========================
# PAGES
# =========================
def page_home():
    st.title("🤖 Agent Studio")
    st.caption("Panel principal: elige un agente o crea uno nuevo.")

    agents = load_agents()
    col1, col2 = st.columns([2, 1], vertical_alignment="top")

    with col1:
        if not agents:
            st.info("Aún no tienes agentes guardados. Crea el primero 🙂")
        else:
            options = {f'{a.get("name","(sin nombre)")} — {a.get("description","")[:60]}': a["id"] for a in agents}
            label = st.selectbox("Agentes", list(options.keys()))
            st.session_state["selected_agent_id"] = options[label]

            selected = next(a for a in agents if a["id"] == st.session_state["selected_agent_id"])
            st.subheader(selected.get("name", "Agente"))
            st.write(selected.get("description", ""))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💬 Abrir chat", use_container_width=True):
                    goto("chat")
                    st.rerun()
            with c2:
                if st.button("✏️ Duplicar (rápido)", use_container_width=True):
                    new_agent = dict(selected)
                    new_agent["id"] = str(uuid.uuid4())
                    new_agent["name"] = f'{selected.get("name","Agente")} (copia)'
                    new_agent["created_at"] = datetime.utcnow().isoformat()
                    upsert_agent(new_agent)
                    st.success("Duplicado. Ya aparece en la lista.")
                    st.rerun()

    with col2:
        st.subheader("Crear")
        if st.button("➕ Crear agente personalizado", use_container_width=True):
            st.session_state["builder_messages"] = []
            st.session_state["agent_draft"] = {"language": "es"}
            st.session_state["agent_ready"] = False
            st.session_state["built_system_prompt"] = ""
            goto("creator")
            st.rerun()

        st.divider()
        st.subheader("Storage")
        st.code(str(AGENTS_PATH), language="text")


def page_creator():
    st.title("🧩 Creador de agentes (con IA)")
    st.caption("La IA te hará preguntas para definir personalidad, límites y estilo. Luego podrás guardarlo.")

    if st.button("⬅️ Volver al Home"):
        goto("home")
        st.rerun()

    if not st.session_state["builder_messages"]:
        first = builder_step("Empecemos. Hazme la primera pregunta para definir el agente.")
        merge_draft(first.get("draft_update", {}))
        st.session_state["agent_ready"] = bool(first.get("done", False))
        st.session_state["built_system_prompt"] = first.get("system_prompt", "") or ""

    with st.expander("📄 Borrador actual", expanded=False):
        st.json(st.session_state.get("agent_draft", {}))

    st.subheader("Conversación con el Builder")
    for msg in st.session_state["builder_messages"]:
        if msg["role"] == "assistant":
            data = extract_json(msg["content"]) or {}
            st.chat_message("assistant").write(data.get("assistant_message", "(sin mensaje)"))
        else:
            st.chat_message("user").write(msg["content"])

    if st.session_state.get("agent_ready") and st.session_state.get("built_system_prompt"):
        st.success("Agente listo para guardar.")
        st.subheader("System prompt generado")
        st.code(st.session_state["built_system_prompt"], language="text")

        colA, colB = st.columns(2)
        with colA:
            if st.button("💾 Guardar agente", use_container_width=True):
                draft = st.session_state.get("agent_draft", {})
                agent = {
                    "id": str(uuid.uuid4()),
                    "name": draft.get("name", "Agente"),
                    "description": draft.get("description", ""),
                    "system_prompt": st.session_state["built_system_prompt"],
                    "model": DEFAULT_CHAT_MODEL,
                    "temperature": 0.7,
                    "created_at": datetime.utcnow().isoformat(),
                }
                upsert_agent(agent)
                st.success("Guardado. Ya aparece en Home.")
                st.session_state["selected_agent_id"] = agent["id"]
                goto("chat")
                st.rerun()
        with colB:
            if st.button("🔄 Reiniciar creador", use_container_width=True):
                st.session_state["builder_messages"] = []
                st.session_state["agent_draft"] = {"language": "es"}
                st.session_state["agent_ready"] = False
                st.session_state["built_system_prompt"] = ""
                st.rerun()
        return

    user_text = st.chat_input("Responde al Builder…")
    if user_text:
        data = builder_step(user_text)
        merge_draft(data.get("draft_update", {}))

        done = bool(data.get("done", False))
        st.session_state["agent_ready"] = done

        if done:
            st.session_state["built_system_prompt"] = data.get("system_prompt", "") or ""
            if not st.session_state["built_system_prompt"]:
                st.session_state["agent_ready"] = False
                st.warning("El builder marcó done pero no entregó system_prompt. Responde una vez más para generarlo.")

        st.rerun()


def page_chat():
    agents = load_agents()

    st.title("💬 Chat")
    if st.button("⬅️ Volver al Home"):
        goto("home")
        st.rerun()

    if not agents:
        st.warning("No hay agentes guardados. Crea uno primero.")
        if st.button("➕ Ir a Crear agente"):
            goto("creator")
            st.rerun()
        return

    options = {f'{a.get("name","(sin nombre)")} — {a.get("description","")[:60]}': a["id"] for a in agents}
    default_id = st.session_state.get("selected_agent_id") or list(options.values())[0]
    labels = list(options.keys())
    ids = list(options.values())

    try:
        default_index = ids.index(default_id)
    except ValueError:
        default_index = 0

    selected_label = st.selectbox("Agente", labels, index=default_index)
    agent_id = options[selected_label]
    st.session_state["selected_agent_id"] = agent_id

    agent = next(a for a in agents if a["id"] == agent_id)

    histories = st.session_state["chat_histories"]
    histories.setdefault(agent_id, [])
    history = histories[agent_id]

    col1, col2, col3 = st.columns([2, 1, 1], vertical_alignment="bottom")
    with col1:
        st.caption(f"Modelo: {agent.get('model', DEFAULT_CHAT_MODEL)} | Temp: {agent.get('temperature', 0.7)}")
    with col2:
        if st.button("🧹 Limpiar chat", use_container_width=True):
            histories[agent_id] = []
            st.rerun()
    with col3:
        with st.popover("⚙️ Ver system prompt"):
            st.code(agent.get("system_prompt", ""), language="text")

    for msg in history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_text = st.chat_input("Escribe tu mensaje…")
    if user_text:
        history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": agent.get("system_prompt", "")}] + history[-30:]
        answer = call_responses(
            model=agent.get("model", DEFAULT_CHAT_MODEL),
            messages=messages,
            temperature=float(agent.get("temperature", 0.7)),
        ).strip()

        if not answer:
            answer = "No obtuve respuesta del modelo. Reintenta."

        history.append({"role": "assistant", "content": answer})
        histories[agent_id] = history
        st.rerun()


# =========================
# MAIN
# =========================
def main():
    init_state()
    check_access()

    page = st.session_state.get("page", "home")

    if page == "home":
        page_home()
    elif page == "creator":
        page_creator()
    elif page == "chat":
        page_chat()
    else:
        goto("home")
        st.rerun()


if __name__ == "__main__":
    main()
