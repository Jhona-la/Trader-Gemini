import time
import threading

class ThreadMonitor:
    def __init__(self):
        self._threads = {}
        self._lock = threading.Lock()

    def register_thread(self, name):
        """
        Registers a thread for monitoring.
        """
        self.update(name, "Registered")

    def update(self, name, status):
        """
        Updates the status and timestamp of a monitored thread.
        """
        with self._lock:
            self._threads[name] = {
                'status': status,
                'last_update': time.time()
            }

    def get_report(self):
        """
        Returns a dictionary with the status and age of all threads.
        """
        now = time.time()
        report = {}
        with self._lock:
            for name, data in self._threads.items():
                age = int(now - data['last_update'])
                report[name] = {
                    'status': data['status'],
                    'last_seen_seconds_ago': age,
                    'is_alive': age < 300 # 5 minutes threshold
                }
        return report

# Global Instance
monitor = ThreadMonitor()
