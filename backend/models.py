from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    documents = relationship("Document", back_populates="owner")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    is_public = Column(Boolean, default=False)
    # SHA-256 του ΠΕΡΙΕΧΟΜΕΝΟΥ. Το file_name δεν αρκεί ως ταυτότητα: το ίδιο
    # paper κατεβασμένο δεύτερη φορά έρχεται ως "1902.03383v1 (1).pdf".
    # nullable: τα έγγραφα που ανέβηκαν πριν τη στήλη (βλ. migrate_add_file_hash).
    file_hash = Column(String, index=True, nullable=True)
    # processing | ready | empty (PDF χωρίς εξαγώγιμο κείμενο) | failed
    status = Column(String, default="processing")
    # Από πόσες σελίδες εξήχθη κείμενο (pdf_report, 23/11/2026). nullable: έγγραφα πριν από τις
    # στήλες (migrate_add_extraction_report.py). pages_empty/pages_scanned: «4,7».
    pages_total = Column(Integer, nullable=True)
    pages_text = Column(Integer, nullable=True)
    pages_empty = Column(String, nullable=True)
    pages_scanned = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="documents")

    # --- ΝΕΟΙ ΠΙΝΑΚΕΣ ΓΙΑ ΤΟ ΙΣΤΟΡΙΚΟ ΣΥΝΟΜΙΛΙΩΝ ---


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="Νέα Συνομιλία")
    user_id = Column(Integer, ForeignKey("users.id"))
    # server_default: την ώρα τη βάζει η ΒΑΣΗ, όχι η διεργασία — με πολλαπλούς
    # workers τα ρολόγια τους δεν είναι εγγυημένα συγχρονισμένα.
    # nullable: οι συνομιλίες που υπάρχουν ΠΡΙΝ τη στήλη δεν έχουν γνωστή ώρα
    # και ΔΕΝ την εφευρίσκουμε (βλ. migrate_add_conversation_created_at.py).
    created_at = Column(DateTime(timezone=True), server_default=func.now(),
                        nullable=True)


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # Θα είναι "user" ή "assistant"
    content = Column(String)
    # Εδώ θα σώζουμε τις πηγές/Citations
    sources = Column(String, nullable=True)
    # Απόσπασμα πηγής ανά πρόταση (κύκλος 2, βήμα 6): JSON, γεμίζει ΜΟΝΟ όταν ο χρήστης πατήσει «Source
    # passages». Σώζεται ώστε το ιστορικό να το δείχνει χωρίς νέα κλήση.
    passages = Column(String, nullable=True)


class Feedback(Base):
    __tablename__ = "feedbacks"
    __table_args__ = (UniqueConstraint("user_id", "message_id", name="uq_feedback_user_message"),)

    id = Column(Integer, primary_key=True, index=True)
    message_id = Column(Integer, ForeignKey("messages.id"))
    is_positive = Column(Boolean)  # True για 👍, False για 👎
    user_id = Column(Integer, ForeignKey("users.id"))
    # Προαιρετικό σχόλιο (κυρίως στα 👎): το «γιατί» δεν άρεσε η απάντηση —
    # τροφοδοτεί τον βρόχο βελτίωσης (failure case -> νέο golden-set case).
    comment = Column(String, nullable=True)



class RateLimit(Base):
    """Μετρητής fixed-window, μοιρασμένος μεταξύ διεργασιών (βλ. rate_limit.py).

    Σύνθετο primary key (bucket, key, window_start): είναι ΚΑΙ το μοναδικό
    constraint πάνω στο οποίο δουλεύει το ON CONFLICT DO UPDATE, άρα η αύξηση
    του μετρητή γίνεται ατομικά μέσα στη βάση — χωρίς race ανάμεσα σε workers.
    """
    __tablename__ = "rate_limits"
    bucket = Column(String, primary_key=True)        # "login" | "chat"
    key = Column(String, primary_key=True)           # IP ή user_id
    window_start = Column(Integer, primary_key=True)  # unix, ευθυγραμμισμένο
    hits = Column(Integer, nullable=False, default=0)

# Στήλες που προστέθηκαν ΜΕΤΑ τη δημιουργία του πίνακα documents (pdf_report, 23/11/2026).
_DOCUMENT_COLUMNS_ADDED = {"pages_total": "INTEGER", "pages_text": "INTEGER",
                           "pages_empty": "VARCHAR", "pages_scanned": "VARCHAR"}


def ensure_document_columns(engine) -> list[str]:
    """Προσθέτει στον πίνακα documents όσες από τις νέες στήλες λείπουν, και επιστρέφει ποιες.

    ΓΙΑΤΙ: το create_all φτιάχνει μόνο νέους πίνακες, δεν προσθέτει στήλες σε υπάρχοντες. Χωρίς αυτό,
    σε βάση που υπήρχε ήδη, κάθε SELECT στα documents σκάει (UndefinedColumn). Το καλούν το main.py στην
    εκκίνηση ΚΑΙ το migrate_add_extraction_report.py: το δεύτερο έσκασε (23/11) γιατί έτρεξε ενώ το
    backend φόρτωνε ακόμα τα μοντέλα και δεν είχε φτάσει να προσθέσει τις στήλες.
    Idempotent, σε Postgres και SQLite."""
    from sqlalchemy import inspect, text

    have = {c["name"] for c in inspect(engine).get_columns("documents")}
    missing = [(name, kind) for name, kind in _DOCUMENT_COLUMNS_ADDED.items() if name not in have]
    if missing:
        with engine.begin() as conn:
            for name, kind in missing:
                conn.execute(text(f"ALTER TABLE documents ADD COLUMN {name} {kind}"))
    return [name for name, _ in missing]


# Στήλες που προστέθηκαν ΜΕΤΑ τη δημιουργία του πίνακα messages (βήμα 6, 24/11/2026).
_MESSAGE_COLUMNS_ADDED = {"passages": "VARCHAR"}


def ensure_message_columns(engine) -> list[str]:
    """Όπως το ensure_document_columns, για τον πίνακα messages. Idempotent, Postgres και SQLite."""
    from sqlalchemy import inspect, text

    have = {c["name"] for c in inspect(engine).get_columns("messages")}
    missing = [(name, kind) for name, kind in _MESSAGE_COLUMNS_ADDED.items() if name not in have]
    if missing:
        with engine.begin() as conn:
            for name, kind in missing:
                conn.execute(text(f"ALTER TABLE messages ADD COLUMN {name} {kind}"))
    return [name for name, _ in missing]
