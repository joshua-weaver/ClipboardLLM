import win32clipboard
import win32con
from typing import Optional, Dict, Any
import io
from PIL import Image
import base64
import struct

class ClipboardReader:
    @staticmethod
    def get_clipboard_content() -> Optional[Dict[str, Any]]:
        """
        Read current clipboard content, supporting multiple formats.
        Returns a dictionary with 'type' and 'content' keys, or None if no supported content.
        """
        try:
            win32clipboard.OpenClipboard()
            
            # Try Unicode text first (most common)
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                return {"type": "text", "content": text}
            
            # Try image formats
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                try:
                    data = win32clipboard.GetClipboardData(win32con.CF_DIB)
                    if data:
                        # The clipboard returns DIB data without the BITMAPFILEHEADER.
                        # We need to add a BITMAPFILEHEADER so that PIL can recognize it as a BMP.
                        header_size = 14
                        # The first 4 bytes of the DIB (BITMAPINFOHEADER) represent its size.
                        biSize = struct.unpack('<I', data[0:4])[0]
                        # The offset to the pixel data is the size of BITMAPFILEHEADER + BITMAPINFOHEADER.
                        bfOffBits = header_size + biSize
                        filesize = len(data) + header_size
                        # BITMAPFILEHEADER:
                        #  - bfType: 2 bytes, must be "BM"
                        #  - bfSize: 4 bytes, total file size
                        #  - bfReserved1 & bfReserved2: 2 bytes each, normally 0
                        #  - bfOffBits: 4 bytes, offset to pixel data
                        file_header = b'BM' + struct.pack('<IHHI', filesize, 0, 0, bfOffBits)
                        bmp_data = file_header + data
                        image = Image.open(io.BytesIO(bmp_data))
                        # Convert image to PNG and encode it as base64
                        buffer = io.BytesIO()
                        image.save(buffer, format="PNG")
                        img_str = base64.b64encode(buffer.getvalue()).decode()
                        return {"type": "image", "content": img_str, "format": "base64"}
                    else:
                        return None
                except Exception as e:
                    print(f"Error processing image data: {e}")
                    return None
            
            # If needed, add support for other formats (e.g., HTML) here.
            return None
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass