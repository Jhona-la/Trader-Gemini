"""
Tests para validar la conciencia micro
"""
import unittest
from core.micro_awareness import MicroAccountAwareness

class TestMicroAwareness(unittest.TestCase):
    def setUp(self):
        self.micro = MicroAccountAwareness()
        
    def test_trade_size_calculation(self):
        size, adjusted = self.micro.calculate_viable_trade_size('BTCUSDT', 50000)
        self.assertGreaterEqual(size * 50000, 5.0)  # Debe cumplir mínimo notional
        
    def test_breakeven_calculation(self):
        breakeven = self.micro.calculate_breakeven_threshold(0.001, 50000)
        self.assertGreater(breakeven, 0.001)  # Debe ser positivo
        
    def test_viability_check(self):
        viable, reason = self.micro.is_trade_viable('BTCUSDT', 50000, 0.02)
        self.assertTrue(viable)  # Debe ser viable con target suficiente

if __name__ == '__main__':
    unittest.main()
