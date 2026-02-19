
import socket
import logging

logger = logging.getLogger("NetworkOptimizer")

def patch_sockets():
    """
     PHASE 31: GLOBAL TCP_NODELAY
     Enforces socket.TCP_NODELAY = 1 on all new sockets.
     This disables Nagle's Algorithm, crucial for HFT latency.
    """
    _original_socket = socket.socket

    class FastSocket(_original_socket):
        def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
            # Resolve defaults manually if needed, but simplest is to pass through
            super().__init__(family, type, proto, fileno)
            
            # Apply TCP_NODELAY if it's a TCP socket
            # Most connections (HTTP, WS) are AF_INET/AF_INET6 and SOCK_STREAM
            if family in (socket.AF_INET, socket.AF_INET6) and type == socket.SOCK_STREAM:
                try:
                    self.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    # logger.debug("⚡ Socket Patcher: TCP_NODELAY set.")
                except Exception as e:
                    # Some platforms/sockets might not support it (e.g. Unix sockets on Windows?)
                    # Just ignore if it fails, don't crash the app
                    pass

    # Apply Monkey Patch
    socket.socket = FastSocket
    logger.info("⚡ [PHASE 31] TCP_NODELAY enforced globally (Nagle's Algorithm Disabled).")
