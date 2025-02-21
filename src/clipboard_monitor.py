import win32gui
import win32con
import win32api
import win32clipboard
import threading
from typing import Callable

class ClipboardMonitor:
    def __init__(self):
        self._running = False
        self._thread = None
        self._hwnd = None
        self._callback = None
        self._last_content = None

    def start(self, callback: Callable) -> None:
        """Start monitoring clipboard changes."""
        if self._running:
            return
        
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()

    def stop(self) -> None:
        """Stop monitoring clipboard changes."""
        self._running = False
        if self._thread:
            self._thread.join()

    def _get_clipboard_content(self):
        """Get current clipboard content."""
        try:
            win32clipboard.OpenClipboard()
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                content = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            else:
                content = None
            win32clipboard.CloseClipboard()
            return content
        except:
            return None

    def _monitor_loop(self) -> None:
        """Monitor clipboard changes in a loop."""
        try:
            while self._running:
                current_content = self._get_clipboard_content()
                if current_content != self._last_content:
                    self._last_content = current_content
                    if self._callback:
                        try:
                            self._callback()
                        except Exception as e:
                            print(f"Error in clipboard callback: {e}")
                threading.Event().wait(0.5)  # Check every 0.5 seconds
        except Exception as e:
            print(f"Error in monitor loop: {e}")
        finally:
            self._running = False