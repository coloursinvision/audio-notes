from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# 1. LOAD CONFIGURATION:
# Fallback to empty dict if .env doesn't exist (e.g., on Streamlit Cloud)
env = dotenv_values(".env")

# 2. SET CONSTANTS
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072
AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"


# 3. OPENAI CLIENT: Using st.secrets or user input
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )
    return transcript.text


# 4. QDRANT CLIENT: Priority given to st.secrets for Cloud deployment
@st.cache_resource
def get_qdrant_client():
    # Use .get() to prevent KeyError if the key is missing in one source
    url = st.secrets.get("qdrant_url") or env.get("qdrant_url")
    api_key = st.secrets.get("QDRANT_API_KEY") or env.get("QDRANT_API_KEY")

    if not url or not api_key:
        st.error("Missing Qdrant configuration! Set qdrant_url and QDRANT_API_KEY in Secrets.")
        st.stop()

    return QdrantClient(url=url, api_key=api_key)


def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )


def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding


def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector=get_embedding(text=note_text),
                payload={"text": note_text},
            )
        ],
    )


def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME, limit=10
        )[0]
        return [{"text": n.payload["text"], "score": None} for n in notes]
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        return [{"text": n.payload["text"], "score": n.score} for n in notes]


# 5. MAIN PAGE CONFIG
st.set_page_config(page_title="Audio Notes", layout="centered")

# 6. OPENAI KEY HANDLING: Priority to st.secrets
if not st.session_state.get("openai_api_key"):
    # Try Cloud secrets first, then local env, then user prompt
    potential_key = st.secrets.get("OPENAI_API_KEY") or env.get("OPENAI_API_KEY")
    if potential_key:
        st.session_state["openai_api_key"] = potential_key
    else:
        st.info("Add your OpenAI API key to use this application")
        st.session_state["openai_api_key"] = st.text_input("API Key", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# 7. SESSION STATE INITIALIZATION
for key in ["note_audio_bytes_md5", "note_audio_bytes", "note_text", "note_audio_text"]:
    if key not in st.session_state:
        st.session_state[key] = None if "bytes" in key else ""

# 8. UI LAYOUT
st.title("Audio Notes")
assure_db_collection_exists()

add_tab, search_tab = st.tabs(["Add note", "Search notes"])

with add_tab:
    note_audio = audiorecorder(
        start_prompt="Record note",
        stop_prompt="Stop recording",
    )
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        if st.button("Transcribe audio"):
            st.session_state["note_audio_text"] = transcribe_audio(
                st.session_state["note_audio_bytes"]
            )

        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area(
                "Edit note", value=st.session_state["note_audio_text"]
            )

        if st.session_state["note_text"] and st.button("Save note"):
            add_note_to_db(note_text=st.session_state["note_text"])
            st.toast("Note saved", icon="🎉")

with search_tab:
    query = st.text_input("Search notes")
    if st.button("Search"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"]:
                    st.markdown(f':violet[{note["score"]}]')
