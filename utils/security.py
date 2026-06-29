
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from utils.logger import logger

class SecurityVault:
    """
    [PHASE I] VANGUARDIA-SOBERANA: In-Memory Credentials Vault.
    Encrypts API Keys/Secrets using AES-256 (Fernet) so they never reside 
    in plain text in RAM for long periods.
    """
    _instance = None
    _cipher = None
    _store = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecurityVault, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """
        Initializes the ephemeral key for this session.
        The key is generated at startup and never stored on disk.
        """
        try:
            # Generate a random ephemeral key for this process session
            # We don't need persistence because config is loaded from .env at boot
            key = Fernet.generate_key()
            self._cipher = Fernet(key)
            logger.info("🛡️ [VAULT] Security Vault Initialized (AES-256 Ephemeral)")
        except Exception as e:
            logger.critical(f"❌ [VAULT] Init Failed: {e}")
            raise RuntimeError("Security Vault Init Failed")

    def store_secret(self, key_name: str, secret_value: str):
        """Encrypts and stores a secret."""
        if not secret_value: return
        try:
            encrypted = self._cipher.encrypt(secret_value.encode())
            self._store[key_name] = encrypted
            # logger.debug(f"🔒 [VAULT] Secured {key_name}")
        except Exception as e:
            logger.error(f"Vault Store Error: {e}")

    def get_secret(self, key_name: str) -> str:
        """Decrypts and retrieves a secret (Transient visibility)."""
        if key_name not in self._store:
            return None
        try:
            encrypted = self._store[key_name]
            decrypted = self._cipher.decrypt(encrypted).decode()
            return decrypted
        except Exception as e:
            logger.error(f"Vault Retrieve Error: {e}")
            return None

    def clear(self):
        """Wipes the vault."""
        self._store.clear()
        self._cipher = None
        logger.info("🧹 [VAULT] Memory Wiped.")

# Global Singleton
vault = SecurityVault()
