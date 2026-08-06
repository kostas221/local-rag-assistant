import os
from datetime import UTC, datetime, timedelta

from dotenv import load_dotenv
from jose import jwt
from passlib.context import CryptContext

# Φορτώνουμε τις μεταβλητές από το .env αρχείο
load_dotenv()

# Τραβάμε το κλειδί με ασφάλεια. Αν δεν βρεθεί, πετάμε error για να μην τρέξει ανασφαλής ο server.
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("CRITICAL: SECRET_KEY environment variable is not set!")

ALGORITHM = "HS256"
# Διάρκεια ζωής token: 1 ημέρα (αντί για ~69 μέρες που ήταν πριν).
# Ένα κλεμμένο token πρέπει να λήγει σύντομα.
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
