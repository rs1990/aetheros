"""Guild-based trust: wallet-signature authentication and DID whitelisting.

Providers never use passwords. On WebSocket connect the orchestrator issues a
one-time nonce; the daemon returns an EIP-191 personal_sign signature over
"grid-auth:<nonce>". The recovered address must (a) match the claimed wallet
and (b) be admitted to the guild — i.e. a recognized Guild Administrator has
signed "guild-admit:<member_wallet>" and that admission is stored in Supabase.
"""

from __future__ import annotations

import logging
import secrets
import time

from eth_account import Account
from eth_account.messages import encode_defunct

from config import settings
from db import store

log = logging.getLogger("grid.auth")

# nonce → issued_at; consumed on first use.
_pending_nonces: dict[str, float] = {}


def issue_nonce() -> str:
    nonce = secrets.token_hex(16)
    _pending_nonces[nonce] = time.monotonic()
    # Opportunistic sweep — the dict stays tiny (one entry per connecting node).
    for n, at in list(_pending_nonces.items()):
        if time.monotonic() - at > settings.NONCE_TTL_SECONDS:
            _pending_nonces.pop(n, None)
    return nonce


def _recover(message: str, signature: str) -> str:
    return Account.recover_message(encode_defunct(text=message), signature=signature)


def verify_handshake(nonce: str, wallet_address: str, signature: str) -> tuple[bool, str]:
    """Validate the signature and guild admission. Returns (ok, reason)."""
    issued = _pending_nonces.pop(nonce, None)
    if issued is None or time.monotonic() - issued > settings.NONCE_TTL_SECONDS:
        return False, "nonce expired or unknown"

    try:
        recovered = _recover(f"grid-auth:{nonce}", signature)
    except Exception as exc:  # malformed signature
        return False, f"signature recovery failed: {exc}"

    if recovered.lower() != wallet_address.lower():
        return False, "signature does not match claimed wallet"

    membership = store.is_admitted(wallet_address.lower()) or store.is_admitted(wallet_address)
    if membership is None:
        return False, "wallet not admitted to any guild"

    # Verify the admission itself: the admin's signature over the member wallet
    # must recover to a recognized guild admin. This closes the spoofed-row
    # vector if the members table is ever writable by a lesser role.
    try:
        admin_recovered = _recover(
            f"guild-admit:{wallet_address.lower()}", membership["admin_signature"]
        )
    except Exception as exc:
        return False, f"admission signature invalid: {exc}"

    if admin_recovered.lower() != membership["admin_address"].lower():
        return False, "admission not signed by listed admin"
    if not store.admin_exists(admin_recovered.lower()) and not store.admin_exists(admin_recovered):
        return False, "admitting admin is not recognized"

    return True, "ok"


def sign_admission(admin_private_key: str, member_wallet: str) -> str:
    """Utility for guild admins: produce the admission signature for a member.

    Run offline by the admin; the result goes into guild_members.admin_signature.
    """
    signed = Account.sign_message(
        encode_defunct(text=f"guild-admit:{member_wallet.lower()}"),
        private_key=admin_private_key,
    )
    return signed.signature.hex()
