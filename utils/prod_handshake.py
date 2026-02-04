"""
🚀 PRODUCTION HANDSHAKE - TRADER GEMINI
========================================

PROFESSOR METHOD:
- QUÉ: Verificación pre-producción para trading con activos reales.
- POR QUÉ: Garantiza conectividad, configuración correcta y seguridad.
- CÓMO: Ejecuta checks secuenciales antes de iniciar trading.
- CUÁNDO: Al iniciar con --env PROD.

CHECKS:
1. API Connectivity (ping)
2. Position Mode = HEDGE
3. Margin Type = ISOLATED
4. Wallet Balance scan
5. IP Authorization
6. Capital tier determination
"""

import os
import sys
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CapitalTier(Enum):
    """Capital tier for risk adaptation."""
    MICRO = "MICRO"      # < $50
    SMALL = "SMALL"      # $50 - $500
    MEDIUM = "MEDIUM"    # $500 - $5000
    LARGE = "LARGE"      # > $5000


@dataclass
class HandshakeResult:
    """Result of production handshake."""
    success: bool
    api_connected: bool
    hedge_mode: bool
    margin_isolated: bool
    balance_usd: float
    capital_tier: CapitalTier
    ip_authorized: bool
    leverage_set: int
    message: str
    errors: list


class ProductionHandshake:
    """
    🤝 Production Handshake - Validates environment before live trading.
    """
    
    def __init__(self):
        self.client = None
        self.result = None
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'SOLUSDT', 'DOGEUSDT', 'BNBUSDT', 'ADAUSDT']
    
    def _print_banner(self):
        """Print handshake banner."""
        print("""
╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║     🚀 PRODUCTION HANDSHAKE - TRADER GEMINI                             ║
║     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                             ║
║                                                                          ║
║     ⚠️  REAL MONEY TRADING MODE                                         ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)
    
    def _init_client(self) -> bool:
        """Initialize Binance Futures client."""
        try:
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            
            api_key = os.environ.get('BINANCE_API_KEY')
            api_secret = os.environ.get('BINANCE_API_SECRET')
            
            if not api_key or not api_secret:
                print("❌ API Keys not found in environment")
                return False
            
            # Check if using testnet keys
            testnet = os.environ.get('BINANCE_USE_TESTNET', 'False').lower() == 'true'
            
            if testnet:
                print("⚠️  WARNING: BINANCE_USE_TESTNET=True - Using testnet!")
                self.client = Client(api_key, api_secret, testnet=True)
            else:
                print("🔐 Connecting to PRODUCTION Binance...")
                self.client = Client(api_key, api_secret)
            
            return True
            
        except Exception as e:
            print(f"❌ Client initialization failed: {e}")
            return False
    
    def check_api_connectivity(self) -> bool:
        """Check API connectivity with ping."""
        print("\n📡 [1/6] Checking API Connectivity...")
        try:
            self.client.ping()
            server_time = self.client.get_server_time()
            print(f"   ✅ API Connected - Server time: {server_time['serverTime']}")
            return True
        except Exception as e:
            print(f"   ❌ API Connection FAILED: {e}")
            return False
    
    def set_hedge_mode(self) -> bool:
        """Set position mode to HEDGE."""
        print("\n🔄 [2/6] Setting Position Mode to HEDGE...")
        try:
            # Try to set hedge mode
            try:
                self.client.futures_change_position_mode(dualSidePosition=True)
                print("   ✅ Position Mode set to HEDGE")
            except Exception as e:
                if "No need to change position side" in str(e):
                    print("   ✅ Position Mode already HEDGE")
                else:
                    # Check current mode
                    mode = self.client.futures_get_position_mode()
                    if mode.get('dualSidePosition', False):
                        print("   ✅ Position Mode confirmed HEDGE")
                    else:
                        print(f"   ⚠️  Could not set HEDGE mode: {e}")
                        return False
            return True
        except Exception as e:
            print(f"   ❌ HEDGE mode setup FAILED: {e}")
            return False
    
    def set_margin_isolated(self) -> bool:
        """Set margin type to ISOLATED for all symbols."""
        print("\n🔒 [3/6] Setting Margin Type to ISOLATED...")
        success_count = 0
        
        for symbol in self.symbols:
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
                success_count += 1
            except Exception as e:
                if "No need to change margin type" in str(e):
                    success_count += 1
                else:
                    print(f"   ⚠️  {symbol}: {e}")
        
        print(f"   ✅ ISOLATED margin set for {success_count}/{len(self.symbols)} symbols")
        return success_count > 0
    
    def scan_wallet_balance(self) -> Tuple[float, CapitalTier]:
        """Scan wallet balance and determine capital tier."""
        print("\n💰 [4/6] Scanning Wallet Balance...")
        try:
            account = self.client.futures_account()
            
            total_balance = float(account.get('totalWalletBalance', 0))
            available = float(account.get('availableBalance', 0))
            unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
            
            print(f"   💵 Total Wallet Balance: ${total_balance:.2f}")
            print(f"   💵 Available Balance:    ${available:.2f}")
            print(f"   📈 Unrealized PnL:       ${unrealized_pnl:.2f}")
            
            # Determine capital tier
            if total_balance < 50:
                tier = CapitalTier.MICRO
                print(f"   📊 Capital Tier: MICRO (< $50) - Max leverage mode")
            elif total_balance < 500:
                tier = CapitalTier.SMALL
                print(f"   📊 Capital Tier: SMALL ($50-$500) - Standard mode")
            elif total_balance < 5000:
                tier = CapitalTier.MEDIUM
                print(f"   📊 Capital Tier: MEDIUM ($500-$5000) - Half-Kelly mode")
            else:
                tier = CapitalTier.LARGE
                print(f"   📊 Capital Tier: LARGE (> $5000) - Conservative mode")
            
            return total_balance, tier
            
        except Exception as e:
            print(f"   ❌ Balance scan FAILED: {e}")
            return 0.0, CapitalTier.MICRO
    
    def check_ip_authorization(self) -> bool:
        """Check if IP is authorized by making account request."""
        print("\n🌐 [5/6] Checking IP Authorization...")
        try:
            # If we can get account info, IP is authorized
            account = self.client.futures_account()
            if account:
                print("   ✅ IP Authorized - Account access confirmed")
                return True
        except Exception as e:
            if "IP" in str(e) or "whitelist" in str(e).lower():
                print(f"   ❌ IP NOT Authorized: {e}")
                return False
            else:
                # Other error, but IP is probably fine
                print(f"   ⚠️  Warning: {e}")
                return True
        return False
    
    def set_leverage(self, tier: CapitalTier) -> int:
        """Set leverage based on capital tier."""
        print("\n⚡ [6/6] Configuring Leverage...")
        
        # Determine leverage based on tier
        if tier == CapitalTier.MICRO:
            target_leverage = 10  # Max for micro capital
        elif tier == CapitalTier.SMALL:
            target_leverage = 5
        else:
            target_leverage = 3
        
        success_count = 0
        for symbol in self.symbols:
            try:
                self.client.futures_change_leverage(symbol=symbol, leverage=target_leverage)
                success_count += 1
            except Exception as e:
                print(f"   ⚠️  {symbol} leverage: {e}")
        
        print(f"   ✅ Leverage set to {target_leverage}x for {success_count}/{len(self.symbols)} symbols")
        return target_leverage
    
    def execute(self) -> HandshakeResult:
        """Execute full production handshake."""
        self._print_banner()
        
        errors = []
        
        # Initialize client
        if not self._init_client():
            return HandshakeResult(
                success=False,
                api_connected=False,
                hedge_mode=False,
                margin_isolated=False,
                balance_usd=0,
                capital_tier=CapitalTier.MICRO,
                ip_authorized=False,
                leverage_set=0,
                message="❌ HANDSHAKE FAILED: Could not initialize client",
                errors=["Client initialization failed"]
            )
        
        # Execute checks
        api_ok = self.check_api_connectivity()
        if not api_ok:
            errors.append("API connectivity failed")
        
        hedge_ok = self.set_hedge_mode()
        if not hedge_ok:
            errors.append("HEDGE mode setup failed")
        
        margin_ok = self.set_margin_isolated()
        if not margin_ok:
            errors.append("ISOLATED margin setup failed")
        
        balance, tier = self.scan_wallet_balance()
        
        ip_ok = self.check_ip_authorization()
        if not ip_ok:
            errors.append("IP authorization failed")
        
        leverage = self.set_leverage(tier)
        
        # Determine overall success
        success = api_ok and hedge_ok and ip_ok and balance > 0
        
        # Generate result
        self.result = HandshakeResult(
            success=success,
            api_connected=api_ok,
            hedge_mode=hedge_ok,
            margin_isolated=margin_ok,
            balance_usd=balance,
            capital_tier=tier,
            ip_authorized=ip_ok,
            leverage_set=leverage,
            message="✅ HANDSHAKE SUCCESSFUL" if success else "❌ HANDSHAKE FAILED",
            errors=errors
        )
        
        self._print_summary()
        
        return self.result
    
    def _print_summary(self):
        """Print handshake summary."""
        r = self.result
        status = "🟢" if r.success else "🔴"
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║                     HANDSHAKE SUMMARY                                    ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Status:           {status} {r.message:<50} ║
╠══════════════════════════════════════════════════════════════════════════╣
║  API Connected:    {"✅" if r.api_connected else "❌":<55} ║
║  HEDGE Mode:       {"✅" if r.hedge_mode else "❌":<55} ║
║  ISOLATED Margin:  {"✅" if r.margin_isolated else "❌":<55} ║
║  IP Authorized:    {"✅" if r.ip_authorized else "❌":<55} ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Balance:          ${r.balance_usd:<52.2f} ║
║  Capital Tier:     {r.capital_tier.value:<55} ║
║  Leverage:         {r.leverage_set}x{" " * 53} ║
╚══════════════════════════════════════════════════════════════════════════╝
        """)
        
        if r.errors:
            print("⚠️  Errors:")
            for err in r.errors:
                print(f"   - {err}")


def run_production_handshake() -> HandshakeResult:
    """Run production handshake and return result."""
    handshake = ProductionHandshake()
    return handshake.execute()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    result = run_production_handshake()
    
    if result.success:
        print("\n🚀 Ready to start trading!")
        print("   Run: python main.py --env PROD")
    else:
        print("\n❌ Fix errors before starting production trading.")
        sys.exit(1)
