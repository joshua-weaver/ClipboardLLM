import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
from datetime import datetime
import win32clipboard
import win32con
import requests
import time
import sv_ttk
from PIL import Image, ImageTk
import io
import base64
import struct
import traceback
import PyPDF2
from docx import Document
import logging
from logger import setup_logger

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

def get_config_path():
    if getattr(sys, 'frozen', False):
        base_dir = os.path.join(os.getenv('APPDATA'), 'ClipboardLLM')
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, 'config.json')

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {str(e)}")
        return f"Error extracting text from DOCX: {str(e)}"

class LLMClient:
    def __init__(self, api_key, provider="openai", model=None):
        self.api_key = api_key
        self.provider = provider.lower()
        self.model = model
        
        # Load config to get system prompt
        try:
            with open(get_config_path(), 'r') as f:
                config = json.load(f)
                self.system_prompt = config.get("system_prompt", "You are a helpful AI assistant.")
        except Exception:
            self.system_prompt = "You are a helpful AI assistant."
        
        if self.provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4"
        elif self.provider == "anthropic":
            self.endpoint = "https://api.anthropic.com/v1/messages"
            self.model = model or "claude-3-sonnet-20240229"
        elif self.provider == "gemini":
            self.endpoint = None
            self.model = model or "gemini-pro-vision"
        else:
            self.endpoint = ""

    def _parse_dib(self, data):
        """Parse DIB header and convert to proper image format"""
        try:
            # Create a BMP header (14 bytes)
            bmp_header = b'BM' + \
                        len(data).to_bytes(4, 'little') + \
                        b'\x00\x00' + \
                        b'\x00\x00' + \
                        b'\x36\x00\x00\x00'  # Standard header size for BMP
            
            # Combine BMP header with DIB data to create a complete BMP
            bmp_data = bmp_header + data
            
            # Use PIL to open the BMP data directly
            image_buffer = io.BytesIO(bmp_data)
            image = Image.open(image_buffer)
            
            # Convert to RGB to ensure consistent color handling
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            raise ValueError(f"Failed to parse DIB data: {str(e)}")

    def _prepare_image_for_ai(self, image):
        """Prepare image in the exact format that will be sent to AI"""
        # Save original image for comparison
        image.save('debug_1_original.png', 'PNG')
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            image.save('debug_2_after_conversion.png', 'PNG')
        
        # Convert to high quality JPEG
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=100, subsampling=0)
        buffer.seek(0)
        jpeg_version = Image.open(buffer)
        jpeg_version.save('debug_3_final_jpeg.jpg', 'JPEG', quality=100, subsampling=0)
        return jpeg_version

    def process_content(self, content: str | bytes, content_type: str = "text") -> str:
        try:
            if content_type == "image":
                print(f"Processing image with provider: {self.provider}")
                
                if isinstance(content, bytes):
                    try:
                        # Convert DIB to PIL Image
                        image = self._parse_dib(content)
                        image.save('debug_0_after_dib_parse.png', 'PNG')
                        
                        # Prepare image in AI format
                        image = self._prepare_image_for_ai(image)
                        
                        # Convert to base64
                        buffer = io.BytesIO()
                        image.save(buffer, format='JPEG', quality=95, subsampling=0)
                        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        print("Successfully converted image to base64")
                    except Exception as e:
                        print(f"Error converting image: {str(e)}")
                        return f"Error converting image: {str(e)}"
                else:
                    print("Content is not bytes, assuming it's already base64")
                    image_b64 = content

                # Process with appropriate provider
                if self.provider == "openai":
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                    data = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_b64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 1024
                    }
                    response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    return response.json()["choices"][0]["message"]["content"].strip()
                
                elif self.provider == "gemini":
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=self.api_key)
                        model = genai.GenerativeModel('gemini-pro-vision')
                        response = model.generate_content(
                            [self.system_prompt, {"mime_type": "image/jpeg", "data": base64.b64decode(image_b64)}]
                        )
                        return response.text
                    except ImportError:
                        return ("Error: google-generativeai package is not installed. "
                                "Please run 'pip install google-generativeai>=0.7.2'.")
                
                else:
                    return "Image processing not supported for this provider"

            # Text processing
            if self.provider == "anthropic":
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                data = {
                    "model": self.model,
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": content}
                    ]
                }
                response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                reply = ""
                if "content" in result:
                    if isinstance(result["content"], list):
                        reply = "".join([item.get("text", "") for item in result["content"] if item.get("type") == "text"])
                    else:
                        reply = result["content"]
                if not reply or not reply.strip():
                    return f"No reply received from Anthropic. Full Response: {result}"
                return reply.strip()
            elif self.provider == "gemini":
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    model = genai.GenerativeModel(self.model)
                    response = model.generate_content(f"{self.system_prompt}\n\nUser: {content}")
                    return response.text
                except ImportError:
                    return ("Error: google-generativeai package is not installed. "
                            "Please run 'pip install google-generativeai>=0.7.2'.")
            else:  # OpenAI default
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": content}],
                    "max_tokens": 1024
                }
                response = requests.post(self.endpoint, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Error processing content: {str(e)}"

class ClipboardMonitor:
    def __init__(self, callback):
        self.callback = callback
        self.running = False
        self.previous_content = None
        self.previous_path = None  # Add this to track the last processed file path

    def start(self):
        self.running = True
        threading.Thread(target=self._monitor, daemon=True).start()

    def stop(self):
        self.running = False

    def _monitor(self):
        while self.running:
            try:
                win32clipboard.OpenClipboard()
                content = None
                content_type = "text"
                current_path = None

                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_HDROP):
                    try:
                        file_list = win32clipboard.GetClipboardData(win32con.CF_HDROP)
                        if file_list:
                            file_path = file_list[0].lower()
                            current_path = file_path
                            if file_path != self.previous_path:
                                if any(file_path.endswith(ext) for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                    print(f"Processing image file: {file_path}")
                                    with Image.open(file_path) as img:
                                        if img.mode != 'RGB':
                                            img = img.convert('RGB')
                                        with io.BytesIO() as output:
                                            img.save(output, format='BMP')
                                            content = output.getvalue()[14:]
                                        content_type = "image"
                                        print("Successfully converted image to DIB format")
                                elif file_path.endswith('.txt'):
                                    print(f"Processing text file: {file_path}")
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    content_type = "text"
                                elif file_path.endswith('.pdf'):
                                    print(f"Processing PDF file: {file_path}")
                                    content = extract_text_from_pdf(file_path)
                                    content_type = "text"
                                elif file_path.endswith('.docx'):
                                    print(f"Processing DOCX file: {file_path}")
                                    content = extract_text_from_docx(file_path)
                                    content_type = "text"
                                else:
                                    print(f"Unsupported file type: {file_path}")
                                    win32clipboard.CloseClipboard()
                                    continue
                    except Exception as e:
                        print(f"Error processing file path: {str(e)}")
                        traceback.print_exc()

                if content is None:
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                        content = win32clipboard.GetClipboardData(win32con.CF_DIB)
                        content_type = "image"
                    elif win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                        content = win32clipboard.GetClipboardData(win32con.CF_TEXT)
                        if content_type == "text":
                            content = content.decode('utf-8', errors='ignore')

                win32clipboard.CloseClipboard()

                if content and (content != self.previous_content or current_path != self.previous_path):
                    self.previous_content = content
                    self.previous_path = current_path
                    self.callback(content, content_type)
            except Exception as e:
                print(f"Clipboard error: {e}")
            time.sleep(1)

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        icon_path = get_resource_path("src/clippy.ico")
        self.root.iconbitmap(icon_path)
        
        # Theme colors from BeardedTheme Coffee-cream
        self.theme = {
            'background': {
                'primary': '#EAE4E1',
                'secondary': '#e3dbd7',
                'paper': '#eee9e7',
                'contrast': '#f4f1f0',
            },
            'text': {
                'primary': '#36221d',
                'secondary': '#a69692',
                'disabled': '#36221d4d',
            },
            'accent': {
                'primary': '#D3694C',
                'secondary': '#009b74',
                'blue': '#008ea4',
                'purple': '#7056c4',
            },
            'border': '#cbbbb4',
        }
        
        # Configure theme styles
        style = ttk.Style()
        
        # Configure common elements
        style.configure('.', 
            background=self.theme['background']['primary'],
            foreground=self.theme['text']['primary'],
            troughcolor=self.theme['background']['secondary'],
            selectbackground=self.theme['accent']['primary'],
            selectforeground=self.theme['text']['primary'],
            bordercolor=self.theme['border'],
            lightcolor=self.theme['background']['paper'],
            darkcolor=self.theme['background']['secondary']
        )
        
        # Frame configurations
        style.configure('TFrame', background=self.theme['background']['primary'])
        style.configure('TLabelframe', 
            background=self.theme['background']['primary'],
            bordercolor=self.theme['border']
        )
        style.configure('TLabelframe.Label', 
            background=self.theme['background']['primary'],
            foreground=self.theme['text']['primary']
        )
        
        # Button configurations
        style.configure('TButton', 
            background=self.theme['accent']['primary'],
            foreground=self.theme['text']['primary'],
            bordercolor=self.theme['border']
        )
        style.map('TButton',
            background=[('active', self.theme['accent']['secondary'])],
            foreground=[('active', self.theme['background']['primary'])]
        )
        
        # Accent button style
        style.configure('Accent.TButton',
            background=self.theme['accent']['primary'],
            foreground=self.theme['text']['primary']
        )
        style.map('Accent.TButton',
            background=[('active', self.theme['accent']['secondary'])],
            foreground=[('active', self.theme['text']['primary'])]
        )
        
        # Entry configurations
        style.configure('TEntry',
            fieldbackground=self.theme['background']['paper'],
            foreground=self.theme['text']['primary'],
            bordercolor=self.theme['border']
        )
        
        # Label configurations
        style.configure('TLabel',
            background=self.theme['background']['primary'],
            foreground=self.theme['text']['primary']
        )
        
        # Checkbutton configurations
        style.configure('TCheckbutton',
            background=self.theme['background']['primary'],
            foreground=self.theme['text']['primary']
        )
        style.map('TCheckbutton',
            background=[('active', self.theme['background']['secondary'])],
            foreground=[('disabled', self.theme['text']['disabled'])]
        )
        
        # Radiobutton configurations
        style.configure('TRadiobutton',
            background=self.theme['background']['primary'],
            foreground=self.theme['text']['primary']
        )
        style.map('TRadiobutton',
            background=[('active', self.theme['background']['secondary'])],
            foreground=[('disabled', self.theme['text']['disabled'])]
        )
        
        # Configure root and menu colors
        self.root.configure(bg=self.theme['background']['primary'])
        
        # Configure menu colors
        menu_config = {
            'background': self.theme['background']['paper'],
            'foreground': self.theme['text']['primary'],
            'activebackground': self.theme['accent']['primary'],
            'activeforeground': self.theme['background']['primary'],
            'selectcolor': self.theme['accent']['primary']
        }
        
        self.root.option_add('*Menu.background', menu_config['background'])
        self.root.option_add('*Menu.foreground', menu_config['foreground'])
        self.root.option_add('*Menu.activeBackground', menu_config['activebackground'])
        self.root.option_add('*Menu.activeForeground', menu_config['activeforeground'])
        self.root.option_add('*Menu.selectColor', menu_config['selectcolor'])
        
        self.root.title("ClipboardLLM")
        self.root.geometry("600x800")

        frame = ttk.Frame(self.root, padding="10", style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True)

        # Chat history with theme colors
        self.chat_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 10),
            background=self.theme['background']['paper'],
            foreground=self.theme['text']['primary'],
            insertbackground=self.theme['text']['primary']
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)

        # Preview frame with theme colors
        self.preview_frame = ttk.LabelFrame(frame, text="Clipboard Content", padding="5")
        
        self.preview_container = ttk.Frame(self.preview_frame)
        self.preview_container.pack(fill=tk.BOTH, expand=True)
        
        self.clipboard_preview = scrolledtext.ScrolledText(
            self.preview_container,
            wrap=tk.WORD,
            height=3,
            font=("Segoe UI", 10),
            background=self.theme['background']['paper'],
            foreground=self.theme['text']['primary'],
            insertbackground=self.theme['text']['primary']
        )
        self.clipboard_preview.pack(fill=tk.X)
        
        self.image_preview = ttk.Label(self.preview_container)

        # Input area with theme colors
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        self.input_text = tk.Text(
            input_frame,
            wrap=tk.WORD,
            height=3,
            font=("Segoe UI", 10),
            background=self.theme['background']['paper'],
            foreground=self.theme['text']['primary'],
            insertbackground=self.theme['text']['primary']
        )
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.input_text.bind("<Return>", self.on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: "break")
        
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.grid(row=0, column=1, sticky="nsew")
        
        self.auto_send = tk.BooleanVar(value=True)
        self.auto_send.trace_add("write", self.on_auto_send_change)
        auto_send_toggle = ttk.Checkbutton(
            input_frame,
            text="Auto Send",
            variable=self.auto_send,
            style="Switch.TCheckbutton"
        )
        auto_send_toggle.grid(row=1, column=0, columnspan=2, pady=(5, 0), sticky="w")
        
        input_frame.columnconfigure(0, weight=1)

        self.status = ttk.Label(
            frame,
            text="Starting...",
            foreground=self.theme['text']['secondary']
        )
        self.status.pack(fill=tk.X, pady=(5, 0))

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        main_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Menu", menu=main_menu)
        main_menu.add_command(label="Settings", command=self.show_settings)
        main_menu.add_command(label="Readme", command=self.show_readme)

        self.monitor = ClipboardMonitor(self.on_clipboard_change)
        self.llm_client = None
        self.current_content_type = "text"
        self._current_text = None
        self._current_image = None
        self.check_config()

        # Update ScrolledText widgets to match theme
        text_config = {
            'background': self.theme['background']['paper'],
            'foreground': self.theme['text']['primary'],
            'insertbackground': self.theme['text']['primary'],
            'selectbackground': self.theme['accent']['primary'],
            'selectforeground': self.theme['background']['paper'],
            'inactiveselectbackground': self.theme['accent']['primary'],
        }
        
        self.chat_text.configure(**text_config)
        self.clipboard_preview.configure(**text_config)
        self.input_text.configure(**text_config)

        # Initialize logger
        self.logger = None

    def on_enter(self, event):
        if not event.state & 0x1:  # Shift key not pressed
            self.send_message()
            return "break"
        return None

    def send_message(self):
        logger = logging.getLogger('ClipboardLLM')
        context = self.input_text.get("1.0", tk.END).strip()
        logger.debug(f"Sending message - Content type: {self.current_content_type}")
        
        if self.current_content_type == "image":
            if context:
                message = f"Image content with additional context:\n{context}"
                logger.debug("Image content with additional context")
            else:
                message = "Image content"
                logger.debug("Image content without context")
            content = self._get_current_image_data()
            self.add_message(message, is_user=True, image_data=content)
        else:
            if self._current_text and context:
                message = f"Clipboard Content:\n{self._current_text}\n\nAdditional Context:\n{context}"
                content = message
                logger.debug("Text content with additional context")
            elif self._current_text:
                message = self._current_text
                content = self._current_text
                logger.debug("Text content without context")
            else:
                message = context
                content = context
                logger.debug("Only context message")
            self.add_message(message, is_user=True)
        
        if not content:
            logger.warning("No content to send")
            return
        
        if not self.llm_client:
            logger.error("LLM client not configured")
            self.add_message("Error: Please configure API key first", is_user=False)
            return
        
        self.input_text.delete("1.0", tk.END)
        self.clipboard_preview.delete("1.0", tk.END)
        self._current_text = None
        if hasattr(self, '_current_image'):
            self._current_image = None
            self.image_preview.pack_forget()
        
        self.status.config(text="Processing message...")
        logger.debug("Starting message processing")
        
        def process():
            try:
                response = self.llm_client.process_content(content, self.current_content_type)
                logger.debug("Successfully received response from LLM")
                self.root.after(0, lambda: self.add_message(response, is_user=False))
                self.root.after(0, lambda: self.status.config(text="Ready"))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.root.after(0, lambda: self.add_message(f"Error: {str(e)}", is_user=False))
                self.root.after(0, lambda: self.status.config(text="Error occurred"))
        
        threading.Thread(target=process, daemon=True).start()

    def _get_current_image_data(self):
        """Return the current image data if it exists"""
        if hasattr(self, '_current_image') and self._current_image is not None:
            print("Returning stored image data")
            return self._current_image
        print("No image data available")
        return None

    def on_clipboard_change(self, content, content_type):
        if not self.llm_client:
            self.add_message("Error: Please configure API key first", is_user=False)
            return
            
        self.current_content_type = content_type
        logger = logging.getLogger('ClipboardLLM')
        logger.debug(f"Clipboard changed - Content type: {content_type}")
        
        if content_type == "image":
            logger.debug("Processing image content")
            self._current_image = content
            self._current_text = None
            try:
                image = self.llm_client._parse_dib(content)
                image = self.llm_client._prepare_image_for_ai(image)
                preview_image = image.copy()
                preview_image.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(preview_image)
                self.clipboard_preview.pack_forget()
                self.image_preview.configure(image=photo)
                self.image_preview.image = photo
                self.image_preview.pack(fill=tk.BOTH, expand=True)
                logger.debug("Successfully processed and displayed image")
                if self.auto_send.get():
                    self.send_message()
                else:
                    self.preview_frame.pack(after=self.chat_text, fill=tk.X, pady=(10, 10))
                    self.status.config(text="Content loaded. Add context and press Send when ready.")
            except Exception as e:
                logger.error(f"Error creating image preview: {e}")
                print(f"Error creating preview: {e}")
                self.clipboard_preview.pack(fill=tk.X)
                self.clipboard_preview.delete("1.0", tk.END)
                self.clipboard_preview.insert(tk.END, "[Image content copied]")
        else:
            logger.debug("Processing text content")
            self._current_text = content
            self._current_image = None
            preview_text = content
            if len(preview_text) > 1000:
                preview_text = preview_text[:1000] + "\n... (truncated)"
                logger.debug("Text content truncated for preview")
            self.image_preview.pack_forget()
            self.clipboard_preview.pack(fill=tk.X)
            self.clipboard_preview.delete("1.0", tk.END)
            self.clipboard_preview.insert(tk.END, preview_text)
            if self.auto_send.get():
                self.send_message()
            else:
                self.preview_frame.pack(after=self.chat_text, fill=tk.X, pady=(10, 10))
                self.status.config(text="Content loaded. Add context and press Send when ready.")

    def on_auto_send_change(self, *args):
        """Handle changes to auto-send toggle."""
        if self.auto_send.get():
            # Hide preview and clear its content when enabling auto-send
            self.preview_frame.pack_forget()
            self.clipboard_preview.delete("1.0", tk.END)
            if hasattr(self, '_current_image'):
                self._current_image = None
        else:
            # Show preview when disabling auto-send
            self.preview_frame.pack(after=self.chat_text, fill=tk.X, pady=(10, 10))

    def add_message(self, message: str, is_user: bool = False, image_data: bytes = None):
        self.chat_text.configure(state=tk.NORMAL)
        if self.chat_text.get("1.0", tk.END).strip():
            self.chat_text.insert(tk.END, "\n\n")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        sender = "You" if is_user else "Assistant"
        self.chat_text.insert(tk.END, f"[{timestamp}] {sender}:\n")
        
        # If there's image data, create and insert thumbnail
        if image_data:
            try:
                # Use LLMClient's parse_dib method to create image
                image = self.llm_client._parse_dib(image_data)
                # Process image exactly as it will be sent to AI
                image = self.llm_client._prepare_image_for_ai(image)
                
                # Create thumbnail
                image.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(image)
                
                # Create a label for the image and insert it into the text widget
                image_label = ttk.Label(self.chat_text, image=photo)
                image_label.image = photo  # Keep reference
                self.chat_text.window_create(tk.END, window=image_label)
                self.chat_text.insert(tk.END, "\n")
            except Exception as e:
                print(f"Error displaying image in chat: {e}")
        
        self.chat_text.insert(tk.END, message)
        self.chat_text.configure(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def check_config(self):
        config_path = get_config_path()
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            active = config.get("selected_provider", "openai")
            providers = config.get("providers", {})
            provider_config = providers.get(active, {})
            api_key = provider_config.get("api_key", "")
            model = provider_config.get("model", "")
            
            # Set up logger based on debug setting
            debug_enabled = config.get("debug_console_enabled", False)
            if debug_enabled:
                self.logger = setup_logger()
            else:
                # Disable logging if debug is off
                logging.getLogger('ClipboardLLM').handlers = []
                logging.getLogger('ClipboardLLM').addHandler(logging.NullHandler())
            
            if not api_key:
                self.show_settings()
            else:
                self.llm_client = LLMClient(api_key, provider=active, model=model)
                self.start_monitoring()
        except Exception:
            self.show_settings()

    def show_settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Settings")
        dialog.geometry("600x600")
        dialog.configure(bg=self.theme['background']['primary'])
        
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Settings headers with theme colors
        ttk.Label(
            frame,
            text="API Configuration",
            font=("Segoe UI", 12, "bold"),
            foreground=self.theme['accent']['primary']
        ).grid(row=0, column=0, columnspan=4, pady=(0, 20), sticky="w")
        
        # Section headers with theme colors
        for text, row in [("Provider", 1), ("API Key", 1), ("Model", 1), ("Primary", 1)]:
            ttk.Label(
                frame,
                text=text,
                font=("Segoe UI", 10, "bold"),
                foreground=self.theme['text']['primary']
            ).grid(
                row=row,
                column=["Provider", "API Key", "Model", "Primary"].index(text),
                pady=(0, 10),
                sticky="w",
                padx=(20 if text != "Provider" else 0, 0)
            )
        
        default_models = {
            "openai": "gpt-4-vision-preview",
            "anthropic": "claude-3-sonnet-20240229",
            "gemini": "gemini-pro-vision"
        }
        
        entries = {}
        model_entries = {}
        selected_provider = tk.StringVar(value="openai")
        
        for i, prov in enumerate(["openai", "anthropic", "gemini"]):
            ttk.Label(frame, text=f"{prov.capitalize()}:").grid(
                row=i+2, column=0, sticky="w", pady=10)
            
            key_entry = ttk.Entry(frame)
            key_entry.grid(row=i+2, column=1, sticky="ew", pady=10, padx=(20, 20))
            entries[prov] = key_entry
            
            model_entry = ttk.Entry(frame)
            model_entry.insert(0, default_models[prov])
            model_entry.grid(row=i+2, column=2, sticky="ew", pady=10, padx=(20, 20))
            model_entries[prov] = model_entry
            
            ttk.Radiobutton(
                frame,
                value=prov.lower(),
                variable=selected_provider
            ).grid(row=i+2, column=3, pady=10, padx=(20, 0))
        
        # System Prompt Section
        ttk.Label(frame, text="System Prompt", font=("Segoe UI", 12, "bold")).grid(
            row=5, column=0, columnspan=4, pady=(30, 10), sticky="w")
        
        system_prompt = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            height=6,
            font=("Segoe UI", 10)
        )
        system_prompt.grid(row=6, column=0, columnspan=4, sticky="ew", pady=(0, 20))
        
        # Debug Console Section
        ttk.Label(frame, text="Debug Settings", font=("Segoe UI", 12, "bold")).grid(
            row=7, column=0, columnspan=4, pady=(10, 10), sticky="w")
        
        debug_frame = ttk.Frame(frame)
        debug_frame.grid(row=8, column=0, columnspan=4, sticky="w", pady=(0, 20))
        
        debug_console_enabled = tk.BooleanVar(value=False)
        debug_toggle = ttk.Checkbutton(
            debug_frame,
            text="Enable Debug Console",
            variable=debug_console_enabled,
            style="Switch.TCheckbutton"
        )
        debug_toggle.pack(side=tk.LEFT, padx=(0, 10))
        
        # Create tooltip for debug toggle
        debug_tooltip = tk.Label(
            debug_frame,
            text="Enable detailed logging to file and console",
            background=self.theme['background']['paper'],
            foreground=self.theme['text']['secondary'],
            relief='solid',
            borderwidth=1
        )
        
        def show_debug_tooltip(event):
            debug_tooltip.place(x=event.widget.winfo_x(), y=event.widget.winfo_y() + 30)
        
        def hide_debug_tooltip(event):
            debug_tooltip.place_forget()
        
        debug_toggle.bind('<Enter>', show_debug_tooltip)
        debug_toggle.bind('<Leave>', hide_debug_tooltip)
        
        # Add button to open logs directory
        def open_logs_dir():
            if getattr(sys, 'frozen', False):
                # If running as exe
                logs_dir = os.path.join(os.getenv('APPDATA'), 'ClipboardLLM', 'logs')
            else:
                # If running from source
                logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            
            # Create directory if it doesn't exist
            os.makedirs(logs_dir, exist_ok=True)
            
            # Open in explorer
            try:
                os.startfile(logs_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open logs directory: {str(e)}")
        
        logs_button = ttk.Button(
            debug_frame,
            text="Open Logs Folder",
            command=open_logs_dir,
            style="Accent.TButton"
        )
        logs_button.pack(side=tk.LEFT)
        
        # Create tooltip for logs button
        logs_tooltip = tk.Label(
            debug_frame,
            text="Open the folder containing debug log files",
            background=self.theme['background']['paper'],
            foreground=self.theme['text']['secondary'],
            relief='solid',
            borderwidth=1
        )
        
        def show_logs_tooltip(event):
            logs_tooltip.place(x=event.widget.winfo_x(), y=event.widget.winfo_y() + 30)
        
        def hide_logs_tooltip(event):
            logs_tooltip.place_forget()
        
        logs_button.bind('<Enter>', show_logs_tooltip)
        logs_button.bind('<Leave>', hide_logs_tooltip)

        # Default system prompt
        default_prompt = ("You are a helpful AI assistant. When provided with text or images, "
                         "analyze them and provide clear, concise, and relevant responses.")
        
        try:
            with open(get_config_path(), 'r') as f:
                config = json.load(f)
            active = config.get("selected_provider", "openai")
            selected_provider.set(active)
            provs = config.get("providers", {})
            for prov in ["openai", "anthropic", "gemini"]:
                if prov in provs:
                    entries[prov].insert(0, provs[prov].get("api_key", ""))
                    if "model" in provs[prov]:
                        model_entries[prov].delete(0, tk.END)
                        model_entries[prov].insert(0, provs[prov]["model"])
            # Load saved system prompt
            saved_prompt = config.get("system_prompt", default_prompt)
            system_prompt.insert("1.0", saved_prompt)
            # Load debug console setting
            debug_console_enabled.set(config.get("debug_console_enabled", False))
        except Exception as e:
            print(f"Error loading config: {e}")
            system_prompt.insert("1.0", default_prompt)

        def save():
            active = selected_provider.get()
            config_data = {
                "selected_provider": active,
                "providers": {
                    prov: {
                        "api_key": entries[prov].get().strip(),
                        "model": model_entries[prov].get().strip()
                    }
                    for prov in ["openai", "anthropic", "gemini"]
                },
                "system_prompt": system_prompt.get("1.0", tk.END).strip(),
                "debug_console_enabled": debug_console_enabled.get()
            }
            with open(get_config_path(), 'w') as f:
                json.dump(config_data, f, indent=4)
            active_key = config_data["providers"][active]["api_key"]
            active_model = config_data["providers"][active]["model"]
            self.llm_client = LLMClient(active_key, provider=active, model=active_model)
            dialog.destroy()
            self.status.config(text="Config saved!")
            self.start_monitoring()
            
            # Update logger based on debug setting
            if debug_console_enabled.get():
                self.logger = setup_logger()
            else:
                # Disable logging if debug is off
                if self.logger:
                    for handler in self.logger.handlers[:]:
                        self.logger.removeHandler(handler)
                logging.getLogger('ClipboardLLM').addHandler(logging.NullHandler())
        
        btn_save = ttk.Button(frame, text="Save", command=save)
        btn_save.grid(row=9, column=0, columnspan=4, pady=(20, 0))
        
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def show_readme(self):
        readme_dialog = tk.Toplevel(self.root)
        readme_dialog.title("Readme")
        readme_dialog.geometry("500x400")
        readme_dialog.transient(self.root)
        readme_dialog.grab_set()
        readme_dialog.configure(bg=self.theme['background']['primary'])
        
        frame = ttk.Frame(readme_dialog, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        readme_text = """ClipboardLLM is an open source application that monitors your clipboard and sends its content (text or images) to various language models (OpenAI, Anthropic, Gemini) for processing.

Features:
- Automatic Clipboard Monitoring: Copy text or images to your clipboard
- Manual Input: Add context to copied content
- Image Support: Process images with compatible models (GPT-4 Vision, Gemini Pro Vision)
- Easy API Configuration: Set up your API keys via the Settings menu

Supported Providers:
- OpenAI: GPT-4 Vision for images and text
- Anthropic: Claude models for text
- Gemini: Pro Vision for images and text

Getting Started:
1. Configure your API key(s) in Settings
2. Select your preferred provider
3. Copy text or images to process

License: MIT License
https://github.com/joshua-weaver/ClipboardLLM

Support:
Venmo: @jshwvr
X: x.com/we4v3r

Enjoy using ClipboardLLM!
"""
        st = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=60, height=20, font=("Segoe UI", 10))
        st.grid(row=0, column=0, sticky="nsew")
        st.insert(tk.END, readme_text)
        st.configure(state=tk.DISABLED)
        
        btn_close = ttk.Button(frame, text="Close", command=readme_dialog.destroy)
        btn_close.grid(row=1, column=0, pady=10)
        
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

    def start_monitoring(self):
        self.status.config(text="Monitoring clipboard...")
        self.monitor.start()
        self.add_message("System: Clipboard monitoring started", is_user=False)

    def start(self):
        try:
            self.root.mainloop()
        finally:
            self.monitor.stop()

if __name__ == "__main__":
    try:
        window = MainWindow()
        window.start()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")