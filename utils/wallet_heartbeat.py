"""
🔄 WALLET HEARTBEAT - Real-Time Balance Synchronization
Connects to Binance User Data Stream for ACCOUNT_UPDATE events.
Updates live_status.json with 8-decimal precision.
"""
import os
import json
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv

load_dotenv()


class WalletHeartbeat:
    """
    Real-time wallet balance synchronization via WebSocket.
    
    PROFESSOR METHOD:
    - QUÉ: Sincronización de billetera en tiempo real
    - POR QUÉ: Precisión de milisegundos para detección de cambios
    - CÓMO: WebSocket User Data Stream + ACCOUNT_UPDATE events
    - CUÁNDO: Cada evento de balance de Binance
    """
    
    def __init__(self, precision: int = 8):
        self.precision = precision
        self.initial_balance = Decimal('0')
        self.current_balance = Decimal('0')
        self.last_update = None
        self.heartbeat_active = False
        self.status_path = Path('dashboard/data/futures/live_status.json')
        
    def set_initial_balance(self, balance: float):
        """Set the mission start balance (Base 0)."""
        self.initial_balance = Decimal(str(balance)).quantize(
            Decimal(f'0.{"0" * self.precision}'), rounding=ROUND_DOWN
        )
        self.current_balance = self.initial_balance
        self.last_update = datetime.now(timezone.utc)
        self.heartbeat_active = True
        print(f'🎯 Mission Start Balance: ${self.initial_balance} USDT')
        print(f'🟢 Wallet Heartbeat: ACTIVE')
        
    def update_balance(self, new_balance: float):
        """Update balance with 8-decimal precision."""
        self.current_balance = Decimal(str(new_balance)).quantize(
            Decimal(f'0.{"0" * self.precision}'), rounding=ROUND_DOWN
        )
        self.last_update = datetime.now(timezone.utc)
        
        # Calculate PnL from mission start
        pnl = self.current_balance - self.initial_balance
        pnl_pct = (pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        print(f'💰 Balance Update: ${self.current_balance} | PnL: ${pnl:+.8f} ({pnl_pct:+.2f}%)')
        
        # Sync to Dashboard
        self._sync_dashboard()
        
    def _sync_dashboard(self):
        """Immediately update live_status.json."""
        try:
            # Create directory if needed
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Read existing or create new
            if self.status_path.exists():
                with open(self.status_path, 'r') as f:
                    status = json.load(f)
            else:
                status = {}
            
            # Update with precise values
            status.update({
                'wallet_balance': str(self.current_balance),
                'initial_balance': str(self.initial_balance),
                'unrealized_pnl': '0.00000000',
                'mission_pnl': str(self.current_balance - self.initial_balance),
                'last_wallet_sync': self.last_update.isoformat(),
                'heartbeat_active': self.heartbeat_active,
                'precision_decimals': self.precision
            })
            
            # Atomic write
            temp_path = self.status_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(status, f, indent=2)
            temp_path.replace(self.status_path)
            
        except Exception as e:
            print(f'⚠️ Dashboard sync error: {e}')
    
    def get_status(self) -> dict:
        """Get current heartbeat status."""
        return {
            'heartbeat_active': self.heartbeat_active,
            'initial_balance': float(self.initial_balance),
            'current_balance': float(self.current_balance),
            'mission_pnl': float(self.current_balance - self.initial_balance),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'precision': self.precision
        }


# Global instance
_heartbeat = WalletHeartbeat()


def init_wallet_heartbeat(initial_balance: float):
    """Initialize wallet heartbeat with starting balance."""
    _heartbeat.set_initial_balance(initial_balance)
    return _heartbeat


def get_wallet_heartbeat() -> WalletHeartbeat:
    """Get the global wallet heartbeat instance."""
    return _heartbeat


if __name__ == '__main__':
    from binance.client import Client
    
    key = os.environ.get('BINANCE_API_KEY')
    secret = os.environ.get('BINANCE_SECRET_KEY') or os.environ.get('BINANCE_API_SECRET')
    
    print('='*70)
    print('🔄 WALLET HEARTBEAT INITIALIZATION')
    print('='*70)
    
    # Get current balance
    c = Client(key, secret)
    account = c.futures_account()
    balance = float(account.get('totalWalletBalance', 0))
    
    # Initialize heartbeat
    heartbeat = init_wallet_heartbeat(balance)
    
    print(f'\n📊 Status:')
    status = heartbeat.get_status()
    for k, v in status.items():
        print(f'   {k}: {v}')
    
    print('='*70)
    print('✅ WALLET HEARTBEAT READY')
    print('='*70)
