import os
import re
import html
import json
import time
import requests
import streamlit as st

# --- Page config MUST be the first Streamlit command ---
st.set_page_config(page_title="Z-AI Platform", page_icon="🔬",
                   layout="wide", initial_sidebar_state="expanded")
API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Logo (SVG) ---
Z_LOGO_HTML = """
<div style="display: flex; justify-content: center; align-items: center; flex-direction: column; padding: 10px;">
    <svg width="64" height="64" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#ff4b4b;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#ff9090;stop-opacity:1" />
            </linearGradient>
        </defs>
        <path d="M25 25 L75 25 L35 75 L75 75" stroke="url(#grad1)" stroke-width="12" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <h3 style="margin-top: 8px; font-family: 'Inter', sans-serif; font-weight: 700; letter-spacing: 3px; color: #9aa0ac;">Z-AI PLATFORM</h3>
</div>
"""

# --- Refined CSS (clean, professional dark theme) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }

    /* Hide Streamlit chrome for a cleaner look */
    #MainMenu, footer, [data-testid="stHeader"] { visibility: hidden; }
    .stAppDeployButton { display: none !important; }
    header { background: transparent !important; height: 0px !important; }
    [data-testid="collapsedControl"] { visibility: visible !important; }

    /* Κεντραρισμένη στήλη συνομιλίας: οι γραμμές σε όλο το πλάτος είναι δυσανάγνωστες. */
    [data-testid="stMainBlockContainer"], .main .block-container {
        max-width: 960px; margin: 0 auto;
    }
    [data-testid="stBottom"] > div { max-width: 960px; margin: 0 auto; }

    /* Chat messages: subtle, clean cards */
    [data-testid="stChatMessage"] {
        background-color: #161922 !important;
        border: 1px solid #232733 !important;
        border-radius: 12px !important;
        padding: 1.1rem 1.3rem !important;
        margin-bottom: 12px !important;
        text-align: left !important;
    }
    /* #1 Readability: να ανασαίνει το κείμενο */
    [data-testid="stChatMessage"] p {
        line-height: 1.65 !important;
        margin-bottom: 0.7rem !important;
    }
    /* #3 Διαφοροποίηση: η ερώτηση του ΧΡΗΣΤΗ διακριτική (χωρίς card),
       ώστε να ξεχωρίζει η ΑΠΑΝΤΗΣΗ (που μένει card). Απαιτεί default user avatar. */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: transparent !important;
        border: none !important;
    }

    /* Buttons */
    .stButton>button { border-radius: 8px; font-weight: 500; transition: all 0.15s ease; }
    .stButton>button:hover { border-color: #ff4b4b; color: #ff4b4b; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #0E1117; }
    ::-webkit-scrollbar-thumb { background: #2a2f3a; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a4150; }
</style>
""", unsafe_allow_html=True)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "token" not in st.session_state:
    st.session_state.token = None
if "current_conv_id" not in st.session_state:
    st.session_state.current_conv_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def set_login_state(token):
    st.session_state.token = token


def logout():
    st.session_state.token = None
    st.session_state.current_conv_id = None
    st.session_state.chat_history = []
    st.rerun()


def change_chat(conv_id):
    st.session_state.current_conv_id = conv_id


def new_chat():
    st.session_state.current_conv_id = None
    st.session_state.chat_history = []


def delete_chat_callback(conv_id, token):
    requests.delete(f"{API_URL}/conversations/{conv_id}",
                    headers={"Authorization": f"Bearer {token}"})
    if st.session_state.current_conv_id == conv_id:
        st.session_state.current_conv_id = None
        st.session_state.chat_history = []


def delete_doc_callback(doc_id, token):
    requests.delete(f"{API_URL}/documents/{doc_id}",
                    headers={"Authorization": f"Bearer {token}"})


def render_sources(sources):
    """Πηγές με preview αποσπάσματος. Συμβατό και με ΠΑΛΙΑ μηνύματα όπου οι
    πηγές ήταν σκέτα strings (backward compatibility)."""
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, src in enumerate(sources):
            if isinstance(src, dict):
                st.caption(f"**[{i+1}]** {src.get('file')} — Page {src.get('page')}")
                preview = src.get("preview")
                if preview:
                    st.markdown(
                        "<div style='border-left:2px solid #ff4b4b; padding:4px 10px; "
                        "margin:2px 0 10px; color:#9aa0ac; font-size:0.85em;'>"
                        f"{html.escape(preview)}</div>",
                        unsafe_allow_html=True)
            else:
                st.caption(f"**[{i+1}]** {src}")


def _library_body(headers, token, disabled=False):
    """Βιβλιοθήκη με LIVE status: το fragment ξανατρέχει ΜΟΝΟ του κάθε 4s, ώστε
    το processing -> ready να εμφανίζεται χωρίς χειροκίνητο refresh. Επικοινωνεί
    με το υπόλοιπο app μέσω st.session_state (όχι κοινές local μεταβλητές).
    disabled=True (όσο streamάρει απάντηση): κλειδώνει τα widgets ώστε ένα κλικ
    να ΜΗΝ διακόπτει το streaming."""
    st.subheader("📂 Library")
    try:
        doc_res = requests.get(f"{API_URL}/documents", headers=headers)
        if doc_res.status_code != 200:
            st.caption("Could not load documents.")
            return
        docs = doc_res.json()
        # Cache ώστε το chat να διαβάσει τα επιλεγμένα αρχεία.
        st.session_state["docs_cache"] = docs
        if not docs:
            st.caption("No documents yet. Upload a PDF below.")
            return

        st.caption("Select sources to search:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select all", use_container_width=True, disabled=disabled):
                for d in docs:
                    st.session_state[f"check_{d['id']}"] = True
        with col2:
            if st.button("Clear all", use_container_width=True, disabled=disabled):
                for d in docs:
                    st.session_state[f"check_{d['id']}"] = False

        for doc in docs:
            display_name = doc['file_name'][:18] + \
                "..." if len(doc['file_name']) > 18 else doc['file_name']
            author_label = "You" if doc['is_mine'] else doc.get(
                'uploader_name', 'Unknown')
            status = doc.get('status', 'ready')
            icon = {"ready": "✅", "processing": "⏳",
                    "failed": "❌"}.get(status, "📄")

            if f"check_{doc['id']}" not in st.session_state:
                st.session_state[f"check_{doc['id']}"] = True

            col1, col2 = st.columns([0.85, 0.15])
            with col1:
                st.checkbox(
                    f"{icon} {display_name} ({author_label})",
                    key=f"check_{doc['id']}",
                    disabled=(disabled or status != "ready"),
                    help=f"{doc['file_name']} — {status}")
            with col2:
                if doc['is_mine']:
                    with st.popover("⋮"):
                        st.button("🗑️", key=f"del_doc_{doc['id']}",
                                  use_container_width=True,
                                  on_click=delete_doc_callback,
                                  args=(doc['id'], token), disabled=disabled)
    except requests.exceptions.RequestException:
        st.warning("⚠️ Could not load documents.")


def render_library(headers, token, generating=False):
    """Wrapper γύρω από το _library_body. Όσο τρέχει απάντηση (generating=True):
      • run_every=None -> παύση auto-refresh (ένα fragment-rerun στη μέση του
        streaming διακόπτει το write_stream)
      • disabled=True -> τα widgets της βιβλιοθήκης (checkbox/Select all/delete)
        κλειδώνουν, ώστε ένα κλικ να ΜΗΝ κόβει την απάντηση.
    Το page-level rerun μετά την απάντηση τα ξανανοίγει."""
    st.fragment(_library_body, run_every=None if generating else "4s")(
        headers, token, disabled=generating)


def app_main(token):
    headers = {"Authorization": f"Bearer {token}"}

    # «Κλείδωμα» όσο τρέχει απάντηση: τα κουμπιά γίνονται disabled ώστε ένα κλικ
    # να ΜΗΝ ακυρώνει το streaming (στο Streamlit κάθε κλικ = rerun -> διακοπή).
    generating = st.session_state.get("pending_question") is not None

    # Φόρτωσε canonical history από το backend — ΟΧΙ όμως όσο streamάρει απάντηση:
    # τότε το backend έχει σώσει την ερώτηση αλλά (ακόμα) όχι την απάντηση, οπότε
    # ένα reload θα έδειχνε μισό ζευγάρι και θα διπλασίαζε το inline-rendered turn.
    if st.session_state.current_conv_id and not generating:
        try:
            msg_res = requests.get(
                f"{API_URL}/conversations/{st.session_state.current_conv_id}/messages", headers=headers)
            if msg_res.status_code == 200:
                st.session_state.chat_history = msg_res.json()
        except requests.exceptions.RequestException:
            st.error("Could not load history. Check that the backend is running.")

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(Z_LOGO_HTML, unsafe_allow_html=True)

        st.button("✨ New Research", use_container_width=True,
                  type="primary", on_click=new_chat, disabled=generating)

        # Answer style -> sent to the backend as "persona"
        # disabled όσο streamάρει: η αλλαγή του = rerun -> κόβει την απάντηση.
        persona = st.selectbox(
            "Answer style",
            ["Researcher", "Educator", "Concise"],
            help="How the assistant phrases its answers",
            key="persona_select",
            disabled=generating)

        st.subheader("Recent Chats")
        try:
            conv_res = requests.get(
                f"{API_URL}/conversations", headers=headers)
            if conv_res.status_code == 200:
                for conv in conv_res.json():
                    display_title = conv['title'][:18] + \
                        "..." if len(conv['title']) > 18 else conv['title']
                    col1, col2 = st.columns([0.85, 0.15])
                    with col1:
                        st.button(f"💬 {display_title}", key=f"conv_{conv['id']}", use_container_width=True, on_click=change_chat, args=(
                            conv['id'],), help=conv['title'], disabled=generating)
                    with col2:
                        with st.popover("⋮"):
                            st.button("🗑️ Delete", key=f"del_conv_{conv['id']}", use_container_width=True, on_click=delete_chat_callback, args=(
                                conv['id'], token), disabled=generating)
        except requests.exceptions.RequestException:
            st.sidebar.error("⚠️ Backend not responding. Chats not loaded.")

        st.divider()

        # --- LIBRARY: live-status fragment (auto-refresh κάθε 4s) ---
        # Όσο streamάρει απάντηση: παύση auto-refresh + disabled widgets, ώστε
        # ΤΙΠΟΤΑ μέσα στη βιβλιοθήκη να μη διακόπτει το streaming με rerun.
        render_library(headers, token, generating=generating)

        st.divider()
        with st.expander("⬆️ Upload Document"):
            uploaded_file = st.file_uploader(
                "PDF", type="pdf", key=f"pdf_uploader_{st.session_state.uploader_key}",
                label_visibility="collapsed", disabled=generating)

            if st.button("Add to Library", use_container_width=True,
                         disabled=generating) and uploaded_file:
                with st.spinner("Uploading & processing..."):
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    upload_headers = {"Authorization": f"Bearer {token}"}

                    response = requests.post(
                        f"{API_URL}/upload", headers=upload_headers, files=files)

                    if response.status_code == 200:
                        st.session_state.uploader_key += 1
                        st.rerun()
                    else:
                        st.error("Something went wrong with the upload.")

        st.button("🚪 Logout", use_container_width=True, on_click=logout, disabled=generating)

    # --- MAIN SCREEN ---
    if not st.session_state.chat_history:
        st.markdown(
            "<h2 style='text-align: center; margin-top: 6vh; font-weight: 600;'>How can I assist your research?</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888; font-size: 1.05em; margin-bottom: 1.5rem;'>Ask questions across your uploaded scientific papers.</p>", unsafe_allow_html=True)

        # Υπόδειξη αντί για generic starters: τα αόριστα prompts κόβονται συχνά
        # από το relevance gate -> καθοδηγούμε σε ΣΥΓΚΕΚΡΙΜΕΝΕΣ ερωτήσεις.
        st.markdown(
            "<p style='text-align:center; color:#8a8f9c; font-size:0.92em;'>"
            "Tip: ask specific questions — e.g. <i>“What is serverless computing?”</i>"
            " or <i>“Ποια είναι τα εμπόδια υιοθέτησης του cloud;”</i></p>",
            unsafe_allow_html=True)

    # --- MESSAGE HISTORY ---
    for index, message in enumerate(st.session_state.chat_history):
        role = message.get("role", "user")
        # default user avatar (όχι custom) ώστε το CSS #3 να το ξεχωρίζει· 🔬 για το AI
        avatar_icon = None if role == "user" else "🔬"

        with st.chat_message(role, avatar=avatar_icon):
            st.write(message.get("content", ""))

            if role == "assistant":
                sources = message.get("sources", [])
                if isinstance(sources, str):
                    try:
                        sources = json.loads(sources)
                    except json.JSONDecodeError:
                        sources = []
                render_sources(sources)

                # Feedback ΜΟΝΟ στις απαντήσεις του AI — στέλνεται πραγματικά στο backend.
                # disabled όσο streamάρει: ένα κλικ εδώ = rerun -> κόβει την απάντηση.
                msg_id = message.get("id")
                feedback = st.feedback("thumbs", key=f"feed_{index}", disabled=generating)
                if feedback is not None and msg_id:
                    # Guard: το st.feedback κρατά την τιμή σε κάθε rerun -> χωρίς
                    # αυτό θα ξαναστέλναμε το ίδιο feedback ξανά και ξανά.
                    sent_key = f"feedback_sent_{msg_id}"
                    if st.session_state.get(sent_key) != feedback:
                        requests.post(
                            f"{API_URL}/feedback", headers=headers,
                            json={"message_id": msg_id, "is_positive": feedback == 1})
                        st.session_state[sent_key] = feedback
                        st.toast("Thanks for your feedback!", icon="📈")

    # Input: ΖΩΓΡΑΦΙΖΕΤΑΙ ΠΡΙΝ το streaming block ώστε το disabled=generating να
    # ισχύει ΚΑΤΑ τη διάρκεια του streaming. (Αν ζωγραφιστεί μετά, όσο το
    # write_stream μπλοκάρει, στην οθόνη μένει η παλιά ΞΕΚΛΕΙΔΩΤΗ έκδοση του input
    # -> μπορείς να γράψεις και να κόψεις την απάντηση.) Το input είναι πάντα
    # καρφιτσωμένο κάτω, οπότε η θέση στον κώδικα δεν αλλάζει την εμφάνιση.
    user_question = st.chat_input("Ask your question...", disabled=generating)
    if user_question and not generating:
        st.session_state.pending_question = user_question
        st.rerun()

    # --- PENDING QUESTION: επεξεργάζεται ΕΔΩ, με το UI «κλειδωμένο» (όλα τα
    # widgets disabled), ώστε ένα κλικ να ΜΗΝ ακυρώνει την απάντηση που τρέχει. ---
    if st.session_state.get("pending_question"):
        q = st.session_state.pending_question

        # IDEMPOTENCY: αν ξαναμπήκαμε εδώ ενώ το stream είχε ήδη ξεκινήσει (το
        # write_stream διακόπηκε από rerun στη μέση), ΜΗΝ ξανα-κάνεις POST — αυτό
        # ήταν που δημιουργούσε διπλό μήνυμα + σκόρπια σειρά. Καθάρισε & ξαναφόρτωσε
        # καθαρά από το backend (περιλαμβάνει όποια μερική απάντηση σώθηκε στο finally).
        if st.session_state.get("pending_inflight"):
            st.session_state.pending_inflight = False
            st.session_state.pending_question = None
            st.rerun()

        # Πρώτη (και μοναδική) αποστολή αυτού του ερωτήματος.
        st.session_state.pending_inflight = True

        with st.chat_message("user"):
            st.write(q)

        if st.session_state.current_conv_id is None:
            with st.spinner("Starting a new session..."):
                conv_res = requests.post(
                    f"{API_URL}/conversations", headers=headers,
                    json={"title": q[:40] + "..."})
                if conv_res.status_code == 200:
                    st.session_state.current_conv_id = conv_res.json()["id"]

        # Επιλεγμένα αρχεία (ready + τσεκαρισμένα) από το fragment της βιβλιοθήκης.
        docs_cache = st.session_state.get("docs_cache", [])
        active_filenames = [
            d["file_name"] for d in docs_cache
            if d.get("status") == "ready"
            and st.session_state.get(f"check_{d['id']}", True)
        ]
        payload = {
            "conversation_id": st.session_state.current_conv_id,
            "question": q,
            "history": st.session_state.chat_history,
            "target_filenames": active_filenames,
            "persona": persona,
        }

        with st.chat_message("assistant", avatar="🔬"):
            message_placeholder = st.empty()
            message_placeholder.markdown("""
                <div style='display:flex; align-items:center; gap:8px; margin-top:-8px;'>
                    <div style='animation: spin 1.4s linear infinite; font-size: 1.1em;'>⏳</div>
                    <div style='color: #888; font-style: italic;'>Thinking…</div>
                </div>
                <style>@keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }</style>
            """, unsafe_allow_html=True)

            sources_data = []

            def stream_backend_response():
                nonlocal sources_data
                first_chunk = True
                try:
                    with requests.post(f"{API_URL}/chat", json=payload, headers=headers,
                                       stream=True, timeout=(10, 180)) as response:
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if not line:
                                    continue
                                data = json.loads(line.decode("utf-8"))
                                if data["type"] == "sources":
                                    sources_data = data["data"]
                                elif data["type"] == "text":
                                    if first_chunk:
                                        message_placeholder.empty()
                                        first_chunk = False
                                    # Smooth streaming: το Gemini στέλνει σε μεγάλα
                                    # μπουρστ -> σπάμε σε λέξεις (lossless: \S+|\s+)
                                    # για ομαλό typewriter effect αντί για «πεταγτό».
                                    for tok in re.findall(r"\S+|\s+", data["data"]):
                                        yield tok
                                        if tok.strip():
                                            time.sleep(0.012)
                        else:
                            message_placeholder.empty()
                            yield f"⚠️ Σφάλμα API ({response.status_code}). Δοκίμασε ξανά σε λίγο."
                except requests.exceptions.Timeout:
                    message_placeholder.empty()
                    yield "⚠️ Το αίτημα άργησε πολύ (timeout). Δοκίμασε ξανά."
                except Exception as e:
                    message_placeholder.empty()
                    yield f"⚠️ Ο server δεν αποκρίνεται: {e}"

            st.write_stream(stream_backend_response())
            render_sources(sources_data)

        # Τελείωσε κανονικά -> ξεκλείδωσε & ξαναφόρτωσε καθαρά από το backend
        # (single source of truth -> καμία διπλοεγγραφή, σωστή σειρά).
        st.session_state.pending_inflight = False
        st.session_state.pending_question = None
        st.rerun()


def auth_screen():
    st.markdown(Z_LOGO_HTML, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.15em; margin-bottom: 40px;'>Academic Research Network</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            log_user = st.text_input("Username")
            log_pass = st.text_input("Password", type="password")
            if st.button("Sign In", use_container_width=True, type="primary"):
                res = requests.post(
                    f"{API_URL}/login", data={"username": log_user, "password": log_pass})
                if res.status_code == 200:
                    set_login_state(res.json()["access_token"])
                    st.rerun()
                elif res.status_code == 429:
                    st.error("Too many attempts. Please try again in a minute.")
                else:
                    st.error("Invalid credentials.")

        with tab2:
            reg_user = st.text_input("New Username")
            reg_email = st.text_input("Email")
            reg_pass = st.text_input("New Password", type="password")

            if st.button("Create Account", use_container_width=True):
                try:
                    res = requests.post(
                        f"{API_URL}/register", json={"username": reg_user, "email": reg_email, "password": reg_pass})

                    if res.status_code in (200, 201):
                        st.success("Account created! You can now log in.")
                    elif res.status_code in (400, 422):
                        st.error(res.json().get(
                            "detail", "Invalid details. (The email or username may already exist.)"))
                    else:
                        st.error(
                            f"Failed to reach the server (error {res.status_code}).")
                except requests.exceptions.RequestException:
                    st.error("The server is unreachable. Please start the backend.")


if st.session_state.token:
    app_main(st.session_state.token)
else:
    auth_screen()
