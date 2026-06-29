import uuid
from typing import Optional
from utils.security import vault

class SecureString:
    """
    🛡️ PHASE I: VANGUARDIA-SOBERANA (AES-256)
    Stores sensitive strings in the Ephemeral Security Vault.
    Replaces simple XOR with Fernet AES-256.
    """
    def __init__(self, secret: str):
        if not secret:
            self._vault_id = None
            return
            
        # Generate a unique ID for this secret
        self._vault_id = str(uuid.uuid4())
        
        # Store in the centralized Vault (AES-256)
        vault.store_secret(self._vault_id, secret)
        
        # Clear local reference hint
        del secret

    def get_unmasked(self) -> Optional[str]:
        """Returns the raw string from Vault."""
        if not self._vault_id:
            return None
        
        return vault.get_secret(self._vault_id)
        
    def __repr__(self):
        return "<SecureString: 🔒 AES-256>"

    def __str__(self):
        return "*****"

    def __del__(self):
        """Cleanup when object is destroyed (Best effort)"""
        # We can't easily remove from vault without explicit cleanup, 
        # but the vault is ephemeral anyway.
        pass

class KeyChain:
    """
    Global secure storage for Runtime secrets.
    """
    _store = {}
    
    @classmethod
    def set(cls, name: str, value: str):
        cls._store[name] = SecureString(value)
        
    @classmethod
    def get(cls, name: str) -> Optional[str]:
        if name in cls._store:
            return cls._store[name].get_unmasked()
        return None
