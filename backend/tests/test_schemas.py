"""Unit tests για τους Pydantic validators (Βήμα 4) στο schemas.py."""
import pytest
from pydantic import ValidationError

from schemas import UserCreate


def test_valid_user_passes():
    u = UserCreate(username="alice", email="alice@example.com",
                   password="secret123")
    assert u.username == "alice"
    assert u.email == "alice@example.com"


def test_short_password_rejected():
    with pytest.raises(ValidationError):
        UserCreate(username="alice", email="a@b.com", password="abc12")


def test_password_without_digit_rejected():
    with pytest.raises(ValidationError):
        UserCreate(username="alice", email="a@b.com", password="onlyletters")


def test_invalid_email_rejected():
    with pytest.raises(ValidationError):
        UserCreate(username="alice", email="not-an-email", password="secret123")


def test_short_username_rejected():
    with pytest.raises(ValidationError):
        UserCreate(username="ab", email="a@b.com", password="secret123")


def test_email_is_normalized_lowercase():
    """Ο validator κάνει το email πεζά + trim."""
    u = UserCreate(username="bob", email="  BOB@Example.COM ",
                   password="secret123")
    assert u.email == "bob@example.com"
