"""
utils/auth.py
--------------
Lightweight username/password authentication backed by IBM COS.

Profiles are stored as JSON objects under the COS path:
  instructor_profiles/<username>.json

Schema:
{
  "username": "prof_smith",
  "password_hash": "<sha256 hex>",
  "full_name": "Dr. Alice Smith",
  "email": "alice@university.edu",
  "role": "instructor",          # or "student"
  "created_at": "<ISO datetime>"
}

No plaintext passwords are persisted — only SHA-256 digests.
"""

import hashlib
import json
import logging
import secrets
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# COS folder prefix for all instructor/user profiles
PROFILE_PREFIX = "instructor_profiles/"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of *password*."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _profile_key(username: str) -> str:
    """Return the COS object key for a given *username*."""
    # Sanitise: keep alphanumeric + underscore + hyphen only
    safe = "".join(c for c in username if c.isalnum() or c in ("_", "-"))
    return f"{PROFILE_PREFIX}{safe}.json"


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def register_user(
    cos,
    username: str,
    password: str,
    full_name: str,
    email: str,
    role: str = "instructor",
) -> tuple[bool, str]:
    """
    Create a new user profile in COS.

    Returns (True, "") on success or (False, error_message) on failure.
    """
    if not username or not password:
        return False, "Username and password are required."

    key = _profile_key(username)

    # Prevent duplicate registrations
    if cos.object_exists(key):
        return False, f"Username '{username}' is already taken."

    profile = {
        "username": username,
        "password_hash": _hash_password(password),
        "full_name": full_name,
        "email": email,
        "role": role,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        cos.put_json(key, profile)
        logger.info("Registered new user: %s (%s)", username, role)
        return True, ""
    except Exception as exc:
        logger.error("Registration failed for %s: %s", username, exc)
        return False, f"Registration failed: {exc}"


def authenticate_user(
    cos,
    username: str,
    password: str,
) -> tuple[bool, Optional[dict]]:
    """
    Verify *username* / *password* against the stored profile.

    Returns (True, profile_dict) on success or (False, None) on failure.
    """
    if not username or not password:
        return False, None

    key = _profile_key(username)
    profile = cos.get_json(key)

    if profile is None:
        logger.warning("Login attempt for unknown user: %s", username)
        return False, None

    if profile.get("password_hash") != _hash_password(password):
        logger.warning("Bad password for user: %s", username)
        return False, None

    logger.info("Authenticated user: %s", username)
    return True, profile


def update_user_profile(cos, username: str, updates: dict) -> tuple[bool, str]:
    """
    Merge *updates* into an existing user profile.
    Prevents overwriting 'password_hash' through this path for safety.
    """
    key = _profile_key(username)
    profile = cos.get_json(key)

    if profile is None:
        return False, f"User '{username}' not found."

    # Never allow bulk password-hash overwrite via this helper
    updates.pop("password_hash", None)
    profile.update(updates)

    try:
        cos.put_json(key, profile)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def list_users(cos, role: Optional[str] = None) -> list[dict]:
    """
    Return all user profiles, optionally filtered by *role*.
    """
    keys = cos.list_prefix(PROFILE_PREFIX)
    users: list[dict] = []
    for key in keys:
        if not key.endswith(".json"):
            continue
        profile = cos.get_json(key)
        if profile and (role is None or profile.get("role") == role):
            users.append(profile)
    return users


def get_user(cos, username: str) -> Optional[dict]:
    """Fetch a single user profile by *username*."""
    return cos.get_json(_profile_key(username))


def reset_password_by_email(
    cos,
    email: str,
    new_password: str,
) -> tuple[bool, str]:
    """
    Reset a user's password identified by their *email* address.

    Searches all profiles for a matching email, then overwrites the
    password_hash. Returns (True, username) on success or (False, error).
    """
    if not email or not new_password:
        return False, "Email and new password are required."

    keys = cos.list_prefix(PROFILE_PREFIX)
    for key in keys:
        if not key.endswith(".json"):
            continue
        profile = cos.get_json(key)
        if profile and profile.get("email", "").lower() == email.strip().lower():
            profile["password_hash"] = _hash_password(new_password)
            profile["updated_at"] = datetime.now(timezone.utc).isoformat()
            try:
                cos.put_json(key, profile)
                logger.info("Password reset for user: %s", profile["username"])
                return True, profile["username"]
            except Exception as exc:
                return False, f"Failed to update password: {exc}"

    return False, "No account found with that email address."
