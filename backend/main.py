import os
import json
import time
import uuid
import sys
import shutil
import logging
from pathlib import Path
from collections import defaultdict, deque
# ΠΡΟΣΘΗΚΗ: Κάναμε import το BackgroundTasks
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, status, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware  # <-- ΝΕΟ IMPORT ΑΣΦΑΛΕΙΑΣ
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, text
from jose import JWTError, jwt
from loguru import logger
import models
import schemas
import database
import security
import ai_core  # <-- ΤΟ ΑΦΗΝΟΥΜΕ ΜΟΝΟ ΕΔΩ (Top-level)

# --- 1. INTERCEPT HANDLER (Ο "Κατάσκοπος" των Logs) ---


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage())


# Καθαρίζουμε τα παλιά logs και βάζουμε το Loguru
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
for logger_name in ("uvicorn.asgi", "uvicorn.access", "uvicorn"):
    logging_logger = logging.getLogger(logger_name)
    logging_logger.handlers = [InterceptHandler()]

# Αρχικοποίηση της Βάσης Δεδομένων
# Αρχικοποίηση της Βάσης Δεδομένων
models.Base.metadata.create_all(bind=database.engine)


def _recover_orphaned_ingests():
    """Αν ο server έπεσε/έκανε restart ενώ ένα έγγραφο ήταν 'processing', το
    BackgroundTask χάθηκε -> το doc θα έμενε για πάντα 'processing'. Στο boot το
    γυρνάμε σε 'failed' ώστε ο χρήστης να ξέρει ότι πρέπει να το ξανανεβάσει."""
    db = database.SessionLocal()
    try:
        stuck = db.query(models.Document).filter(
            models.Document.status == "processing").all()
        for doc in stuck:
            doc.status = "failed"
        if stuck:
            db.commit()
            logger.warning(
                f"Recovery: {len(stuck)} έγγραφα 'processing' -> 'failed' "
                "(χάθηκαν σε restart του server).")
    finally:
        db.close()


_recover_orphaned_ingests()

app = FastAPI(title="Z-AI Platform API")

# --- 2. CORS MIDDLEWARE (Ασφάλεια Δικτύου) ---
# Επιτρέπει στο Streamlit (ή σε οποιοδήποτε άλλο frontend στο μέλλον) να μιλάει με το API σου
# CORS origins από env (comma-separated)· default το τοπικό Streamlit. Στο deploy
# βάλε το domain του frontend, π.χ. ALLOWED_ORIGINS="https://rag.sxoli.gr".
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. MIDDLEWARE ΠΑΡΑΤΗΡΗΣΙΜΟΤΗΤΑΣ ---


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    logger.info(
        f"Εκκίνηση Αιτήματος: {request.method} {request.url.path} [ID: {request_id}]")

    response = await call_next(request)

    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id

    logger.info(
        f"Ολοκλήρωση: {request.method} {request.url.path} - Status: {response.status_code} - Χρόνος: {process_time:.4f}s [ID: {request_id}]")

    return response

# --- 4. GLOBAL EXCEPTION HANDLERS ---


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(
        f"Ακυρώθηκε λόγω κακών δεδομένων στο {request.url.path}: {exc}")
    # Πάρε το πρώτο ανθρώπινο μήνυμα ώστε να το δει ο χρήστης (π.χ. "κωδικός μικρός")
    try:
        detail = exc.errors()[0].get("msg", "Μη έγκυρα δεδομένα εισόδου.")
    except Exception:
        detail = "Μη έγκυρα δεδομένα εισόδου."
    return JSONResponse(status_code=422, content={"detail": detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.critical(f"ΚΡΙΣΙΜΟ ΣΦΑΛΜΑ στο {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Προέκυψε εσωτερικό σφάλμα. Η ομάδα έχει ενημερωθεί."}
    )

# Φάκελος για την αποθήκευση των PDF (Συμβατό με το Docker Mount)
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Readiness probe: ελαφρύ, ελέγχει ΚΑΙ τη σύνδεση με τη βάση.
    Επιστρέφει 503 αν η βάση δεν απαντά -> το orchestrator ξέρει να μην
    στείλει traffic σε αυτό το instance."""
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check ΑΠΕΤΥΧΕ: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "database": "disconnected"})


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, security.SECRET_KEY,
                             algorithms=[security.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(models.User).filter(
        models.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user_by_username = db.query(models.User).filter(
        models.User.username == user.username).first()
    if db_user_by_username:
        raise HTTPException(
            status_code=400, detail="Το Username υπάρχει ήδη.")

    db_user_by_email = db.query(models.User).filter(
        models.User.email == user.email).first()
    if db_user_by_email:
        raise HTTPException(
            status_code=400, detail="Το Email υπάρχει ήδη.")

    hashed_password = security.get_password_hash(user.password)
    new_user = models.User(username=user.username,
                           email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully"}


# --- Rate limiting στο login (anti brute-force) ---
# Απλό in-memory fixed-window: max N προσπάθειες ανά IP σε X δευτερόλεπτα.
LOGIN_MAX_ATTEMPTS = 5
LOGIN_WINDOW_SEC = 60
_login_attempts = defaultdict(deque)


def check_login_rate_limit(ip: str):
    now = time.time()
    # Καθάρισε ληγμένα IPs ώστε το dict να μη μεγαλώνει επ' άπειρον (memory leak):
    # χωρίς αυτό, κάθε IP που μπήκε ποτέ έμενε για πάντα στη μνήμη.
    if len(_login_attempts) > 1000:
        for k in [k for k, dq in _login_attempts.items()
                  if not dq or now - dq[-1] > LOGIN_WINDOW_SEC]:
            del _login_attempts[k]
    attempts = _login_attempts[ip]
    # Πέτα τις προσπάθειες που βγήκαν εκτός του χρονικού παραθύρου
    while attempts and now - attempts[0] > LOGIN_WINDOW_SEC:
        attempts.popleft()
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Πολλές προσπάθειες σύνδεσης. Δοκίμασε ξανά σε ένα λεπτό.")
    attempts.append(now)


@app.post("/login")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    check_login_rate_limit(request.client.host)
    user = db.query(models.User).filter(
        models.User.username == form_data.username).first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401, detail="Incorrect username or password")

    access_token = security.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


def process_document(doc_id: int, file_path: str, filename: str, user_id: int):
    """Τρέχει στο παρασκήνιο: κάνει ingest και ενημερώνει το status του εγγράφου."""
    db = database.SessionLocal()
    try:
        ai_core.ingest_pdf(file_path, filename, user_id, doc_id=doc_id)
        new_status = "ready"
    except Exception as e:
        logger.error(f"Ingest απέτυχε για '{filename}': {e}")
        new_status = "failed"
    try:
        doc = db.query(models.Document).filter(
            models.Document.id == doc_id).first()
        if doc:
            doc.status = new_status
            db.commit()
    finally:
        db.close()

# ΠΡΟΣΘΗΚΗ: Ασύγχρονη επεξεργασία αρχείων


@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # --- ΑΣΦΑΛΕΙΑ: καθαρισμός ονόματος (απέναντι σε path traversal) ---
    safe_name = Path(file.filename or "").name  # πετάει ../ και διαδρομές
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Επιτρέπονται μόνο αρχεία PDF.")

    # Μοναδικό όνομα στον δίσκο ώστε δύο αρχεία ίδιου ονόματος να μη συγκρούονται
    disk_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = os.path.join(UPLOAD_DIR, disk_name)

    # Γράψιμο σε streaming (σταθερή μνήμη ακόμα και για μεγάλα PDF)
        # Γράψιμο σε streaming ΜΕ όριο μεγέθους: χωρίς αυτό ένα τεράστιο "PDF"
    # γεμίζει τον δίσκο πριν καν φτάσει στο ingest.
    MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB
    written = 0
    too_big = False
    with open(file_path, "wb") as buffer:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                too_big = True
                break
            buffer.write(chunk)
    if too_big:
        os.remove(file_path)
        raise HTTPException(status_code=413,
                            detail="Το αρχείο ξεπερνά το όριο των 50MB.")

    new_doc = models.Document(file_name=safe_name, file_path=file_path,
                              user_id=current_user.id, status="processing")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)  # για να πάρουμε το new_doc.id

    # ΑΝΑΘΕΣΗ ΣΤΟ ΠΑΡΑΣΚΗΝΙΟ (ενημερώνει και το status στο τέλος)
    background_tasks.add_task(
        process_document, new_doc.id, file_path, safe_name, current_user.id)

    # Ο χρήστης λαμβάνει απάντηση αμέσως
    return {"message": "Το αρχείο ανέβηκε επιτυχώς. Η επεξεργασία του ξεκίνησε στο παρασκήνιο."}


@app.get("/documents")
def get_documents(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Βλέπω ΜΟΝΟ τα δικά μου έγγραφα + όσα είναι public (όχι τα ιδιωτικά άλλων).
    # joinedload: ένα query με JOIN αντί για N+1 ξεχωριστά queries.
    all_docs = (
        db.query(models.Document)
        .options(joinedload(models.Document.owner))
        .filter(or_(models.Document.user_id == current_user.id,
                    models.Document.is_public == True))
        .all()
    )

    return [
        {
            "id": doc.id,
            "file_name": doc.file_name,
            "is_mine": doc.user_id == current_user.id,
            "uploader_name": doc.owner.username if doc.owner else "Unknown",
            "status": doc.status,
        }
        for doc in all_docs
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    doc = db.query(models.Document).filter(models.Document.id ==
                                           doc_id, models.Document.user_id == current_user.id).first()
    if not doc:
        raise HTTPException(
            status_code=404, detail="Document not found or not authorized")

    # Διαγράφουμε ΜΟΝΟ τα chunks αυτού του χρήστη (όχι ομώνυμων άλλων).
    ai_core.delete_file_from_db(doc.file_name, user_id=current_user.id, doc_id=doc.id)
    # Σβήσε και το PDF από τον δίσκο (αλλιώς το uploaded_docs/ μεγαλώνει για πάντα).
    try:
        if doc.file_path and os.path.exists(doc.file_path):
            os.remove(doc.file_path)
    except OSError as e:
        logger.warning(f"Δεν διαγράφηκε το αρχείο '{doc.file_path}': {e}")
    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}


@app.post("/conversations")
def create_conversation(conv: schemas.ConversationCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_conv = models.Conversation(title=conv.title, user_id=current_user.id)
    db.add(new_conv)
    db.commit()
    db.refresh(new_conv)
    return new_conv


@app.get("/conversations")
def get_conversations(current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(models.Conversation).filter(models.Conversation.user_id == current_user.id).order_by(models.Conversation.id.desc()).all()


@app.get("/conversations/{conversation_id}/messages")
def get_messages(conversation_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter(models.Conversation.id ==
                                                conversation_id, models.Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # ORDER BY id: χωρίς αυτό η Postgres επιστρέφει σε ΑΠΡΟΣΔΙΟΡΙΣΤΗ σειρά —
    # τα μηνύματα φαίνονταν σκόρπια (π.χ. Q,Q,A,A) παρότι σώζονταν σωστά (Q,A,Q,A).
    return db.query(models.Message).filter(
        models.Message.conversation_id == conversation_id).order_by(models.Message.id).all()


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(models.Conversation).filter(models.Conversation.id ==
                                                conversation_id, models.Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Καθάρισε ΚΑΙ τα feedback των μηνυμάτων, αλλιώς μένουν ορφανά (το Feedback
    # δείχνει σε message_id που πρόκειται να σβηστεί).
    msg_ids = [m.id for m in db.query(models.Message.id).filter(
        models.Message.conversation_id == conversation_id).all()]
    if msg_ids:
        db.query(models.Feedback).filter(
            models.Feedback.message_id.in_(msg_ids)).delete(synchronize_session=False)

    db.query(models.Message).filter(
        models.Message.conversation_id == conversation_id).delete()

    db.delete(conv)
    db.commit()
    return {"message": "Conversation completely deleted"}


@app.post("/chat")
async def chat_with_ai(query: schemas.ChatMessage, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    # ΑΣΦΑΛΕΙΑ: επιβεβαίωσε ότι το conversation ανήκει στον χρήστη
    conv = db.query(models.Conversation).filter(
        models.Conversation.id == query.conversation_id,
        models.Conversation.user_id == current_user.id).first()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    user_msg = models.Message(
        conversation_id=query.conversation_id, role="user", content=query.question)
    db.add(user_msg)
    db.commit()

    async def event_stream():
        full_answer = ""
        final_sources = []

        try:
            async for packet in ai_core.ask_ai(question=query.question, target_filenames=query.target_filenames, history=query.history, user_id=current_user.id, persona=query.persona):

                if packet["type"] == "sources":
                    final_sources = packet["data"]
                    yield f"{json.dumps({'type': 'sources', 'data': final_sources})}\n"

                elif packet["type"] == "text":
                    full_answer += packet["data"]
                    yield f"{json.dumps({'type': 'text', 'data': packet['data']})}\n"
        finally:
            # Τρέχει ΚΑΙ αν ο χρήστης κλείσει τη σελίδα στη μέση του streaming:
            # σώζουμε ό,τι παρήχθη, αλλιώς η απάντηση εξαφανίζεται από το ιστορικό.
            if full_answer:
                ai_msg = models.Message(
                    conversation_id=query.conversation_id,
                    role="assistant",
                    content=full_answer,
                    sources=json.dumps(final_sources)
                )
                db.add(ai_msg)
                db.commit()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/feedback")
def submit_feedback(fb: schemas.FeedbackCreate, current_user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_fb = models.Feedback(
        message_id=fb.message_id,
        is_positive=fb.is_positive,
        user_id=current_user.id
    )
    db.add(new_fb)
    db.commit()
    return {"status": "success", "message": "Feedback recorded"}
