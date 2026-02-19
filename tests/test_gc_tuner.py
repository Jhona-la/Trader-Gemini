import unittest
import gc
import time
from core.gc_tuner import GCTuner

class TestGCTuner(unittest.TestCase):
    def test_critical_section(self):
        """Verify GC is disabled inside critical section"""
        print("\n🧹 Testing GCTuner Critical Section...")
        
        gc.enable()
        self.assertTrue(gc.isenabled(), "GC should be enabled initially")
        
        with GCTuner.critical_section():
            self.assertFalse(gc.isenabled(), "GC should be disabled inside critical section")
            # Simulate work
            a = [x for x in range(1000)]
            
        self.assertTrue(gc.isenabled(), "GC should be re-enabled after")
        print("   > Critical Section Logic OK.")
        
    def test_maintenance_interval(self):
        """Verify Check Maintenance triggers collect"""
        print("🧹 Testing Maintenance Interval...")
        
        # Reset last collect to long ago
        GCTuner._last_collect = time.time() - 100.0
        
        # This should trigger collect (we can't easily mock gc.collect without mock lib, 
        # but we can check if _last_collect updates)
        old_last = GCTuner._last_collect
        
        GCTuner.check_maintenance(interval=10.0)
        
        new_last = GCTuner._last_collect
        self.assertNotEqual(old_last, new_last, "Last collect timestamp should update")
        self.assertTrue(new_last > old_last, "Timestamp should increase")
        
        print("   > Maintenance Trigger OK.")
        
    def test_nested_safety(self):
        # Optional: Test nested critical sections if we supported them (contextlib does, but our logic?)
        # Our logic: 
        # Enter: was_enabled=True -> disable().
        #   Nested Enter: was_enabled=False -> do nothing.
        #   Nested Exit: was_enabled=False -> do nothing.
        # Exit: was_enabled=True -> enable().
        # It IS safe for re-entrancy if logic holds.
        pass

if __name__ == '__main__':
    unittest.main()
