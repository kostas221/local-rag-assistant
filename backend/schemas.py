import re
from typing import Any

from pydantic import BaseModel, validator

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Σχέδιο για εγγραφή χρήστη


class UserCreate(BaseModel):
    username: str
    email: str
    password: str

    @validator("username")
    def username_valid(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if not re.match(r"^[A-Za-z0-9_.-]+$", v):
            raise ValueError(
                "Username may only contain letters, numbers and _ . -")
        return v

    @validator("email")
    def email_valid(cls, v):
        v = v.strip().lower()
        if not EMAIL_RE.match(v):
            raise ValueError("Invalid email format.")
        return v

    @validator("password")
    def password_strong(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit.")
        return v


class ConversationCreate(BaseModel):
    title: str

# Σχέδιο για το μήνυμα που στέλνει ο χρήστης στο AI


class ChatMessage(BaseModel):
    conversation_id: int
    question: str
    target_filenames: list = []
    persona: str = "Researcher"  # The AI answer style
    # <--- Η ΠΡΟΣΘΗΚΗ ΜΑΣ: Δέχεται λίστα από λεξικά (τη μνήμη)
    history: list[dict[str, Any]] = []

# Σχέδιο για το 👍 / 👎 feedback


class FeedbackCreate(BaseModel):
    message_id: int
    is_positive: bool
    comment: str | None = None

    @validator("comment")
    def _trim_comment(cls, v):
        # Κόβουμε κενά + βάζουμε καπάκι μεγέθους: το πεδίο είναι ελεύθερο κείμενο
        # από χρήστες -> χωρίς όριο, ένα copy-paste βιβλίο γίνεται γραμμή στη βάση.
        if v is None:
            return None
        v = v.strip()
        return v[:1000] if v else None
