"""Unit tests για το security.py — hashing κωδικών & JWT tokens."""
from jose import jwt

from security import ALGORITHM, SECRET_KEY, create_access_token, get_password_hash, verify_password


def test_password_hash_is_not_plaintext():
    """Ο κωδικός ΔΕΝ αποθηκεύεται ποτέ σε καθαρή μορφή."""
    hashed = get_password_hash("MySecret123")
    assert hashed != "MySecret123"
    assert hashed.startswith("$2")  # bcrypt prefix


def test_verify_password_roundtrip():
    """Σωστός κωδικός -> True, λάθος κωδικός -> False."""
    hashed = get_password_hash("MySecret123")
    assert verify_password("MySecret123", hashed) is True
    assert verify_password("wrongpassword", hashed) is False


def test_access_token_contains_subject_and_expiry():
    """Το JWT πρέπει να κουβαλάει το 'sub' (χρήστη) και ημερομηνία λήξης."""
    token = create_access_token({"sub": "alice"})
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "alice"
    assert "exp" in payload
