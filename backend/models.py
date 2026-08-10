from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
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
    # processing | ready | failed
    status = Column(String, default="processing")
    user_id = Column(Integer, ForeignKey("users.id"))

    owner = relationship("User", back_populates="documents")

    # --- ΝΕΟΙ ΠΙΝΑΚΕΣ ΓΙΑ ΤΟ ΙΣΤΟΡΙΚΟ ΣΥΝΟΜΙΛΙΩΝ ---


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, default="Νέα Συνομιλία")
    user_id = Column(Integer, ForeignKey("users.id"))


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # Θα είναι "user" ή "assistant"
    content = Column(String)
    # Εδώ θα σώζουμε τις πηγές/Citations
    sources = Column(String, nullable=True)


class Feedback(Base):
    __tablename__ = "feedbacks"
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