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
                current_path = None  # Track the current file path

                # Check for file paths (CF_HDROP for file drops)
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_HDROP):
                    try:
                        file_list = win32clipboard.GetClipboardData(win32con.CF_HDROP)
                        
                        if file_list:
                            file_path = file_list[0].lower()
                            current_path = file_path  # Store the current path
                            
                            # Only process if it's a different file from the last one
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
                    except Exception as e:
                        print(f"Error processing file path: {str(e)}")
                        traceback.print_exc()

                # If no file was processed, check for image or text data
                if content is None:
                    if win32clipboard.IsClipboardFormatAvailable(win32con.CF_DIB):
                        content = win32clipboard.GetClipboardData(win32con.CF_DIB)
                        content_type = "image"
                    elif win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                        content = win32clipboard.GetClipboardData(win32con.CF_TEXT)
                        if content_type == "text":
                            content = content.decode('utf-8', errors='ignore')

                win32clipboard.CloseClipboard()

                # Update both content and path if we processed something new
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
        sv_ttk.set_theme("dark")
        self.root.title("ClipboardLLM")
        self.root.geometry("600x800")

        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # Chat history
        self.chat_text = scrolledtext.ScrolledText(
            frame,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Segoe UI", 10),
            background="#1e1e1e",
            foreground="#ffffff",
            insertbackground="#ffffff"
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)

        # Add clipboard preview box (initially hidden)
        self.preview_frame = ttk.LabelFrame(frame, text="Clipboard Content", padding="5")
        
        # Create a frame to hold both text and image previews
        self.preview_container = ttk.Frame(self.preview_frame)
        self.preview_container.pack(fill=tk.BOTH, expand=True)
        
        self.clipboard_preview = scrolledtext.ScrolledText(
            self.preview_container,
            wrap=tk.WORD,
            height=3,
            font=("Segoe UI", 10),
            background="#1e1e1e",
            foreground="#ffffff",
            insertbackground="#ffffff"
        )
        self.clipboard_preview.pack(fill=tk.X)
        
        # Image preview label (hidden initially)
        self.image_preview = ttk.Label(self.preview_container)

        # Input area
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, pady=(5, 0))
        self.input_text = tk.Text(
            input_frame,
            wrap=tk.WORD,
            height=3,
            font=("Segoe UI", 10),
            background="#1e1e1e",
            foreground="#ffffff",
            insertbackground="#ffffff"
        )
        self.input_text.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.input_text.bind("<Return>", self.on_enter)
        self.input_text.bind("<Shift-Return>", lambda e: "break")
        send_button = ttk.Button(input_frame, text="Send", command=self.send_message)
        send_button.grid(row=0, column=1, sticky="nsew")
        # Auto-send toggle with callback
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

        self.status = ttk.Label(frame, text="Starting...")
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
        self.check_config()

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

    def on_enter(self, event):
        if not event.state & 0x1:  # Shift key not pressed
            self.send_message()
            return "break"
        return None

    def send_message(self):
        """Modified to include clipboard content when sending."""
        context = self.input_text.get("1.0", tk.END).strip()
        clipboard_content = self.clipboard_preview.get("1.0", tk.END).strip()
        
        # Handle different content types
        if self.current_content_type == "image":
            if context and not self.auto_send.get():
                message = f"Image content with additional context:\n{context}"
            else:
                message = "Image content"
            content = self._get_current_image_data()
            # Add the image to the chat with the message
            self.add_message(message, is_user=True, image_data=content)
        else:
            # Text content handling
            if clipboard_content and context and not self.auto_send.get():
                # Only combine content and context when auto-send is off
                message = f"Clipboard Content:\n{clipboard_content}\n\nAdditional Context:\n{context}"
                content = message
            elif clipboard_content:
                message = clipboard_content
                content = clipboard_content
            else:
                message = context
                content = context
            # Add text message without image
            self.add_message(message, is_user=True)

        if not content:
            return
        
        if not self.llm_client:
            self.add_message("Error: Please configure API key first", is_user=False)
            return

        # Clear both input boxes
        self.input_text.delete("1.0", tk.END)
        self.clipboard_preview.delete("1.0", tk.END)
        if hasattr(self, '_current_image'):
            self._current_image = None
            self.image_preview.pack_forget()
        
        self.status.config(text="Processing message...")
        
        def process():
            response = self.llm_client.process_content(content, self.current_content_type)
            self.root.after(0, lambda: self.add_message(response, is_user=False))
            self.root.after(0, lambda: self.status.config(text="Ready"))
        
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
        
        if content_type == "image":
            # Store the raw image data immediately
            self._current_image = content
            
            try:
                # Create preview using LLMClient's parse_dib method
                image = self.llm_client._parse_dib(content)
                # Process image exactly as it will be sent to AI
                image = self.llm_client._prepare_image_for_ai(image)
                
                # Create preview thumbnail
                preview_image = image.copy()
                preview_image.thumbnail((200, 200))
                photo = ImageTk.PhotoImage(preview_image)
                
                # Update preview
                self.clipboard_preview.pack_forget()
                self.image_preview.configure(image=photo)
                self.image_preview.image = photo  # Keep reference
                self.image_preview.pack(fill=tk.BOTH, expand=True)
                
                if self.auto_send.get():
                    # Automatically send if auto-send is enabled
                    self.send_message()
                else:
                    # Show the preview frame if auto-send is disabled
                    self.preview_frame.pack(after=self.chat_text, fill=tk.X, pady=(10, 10))
                    self.status.config(text="Content loaded. Add context and press Send when ready.")
                
            except Exception as e:
                print(f"Error creating preview: {e}")
                self.clipboard_preview.pack(fill=tk.X)
                self.clipboard_preview.delete("1.0", tk.END)
                self.clipboard_preview.insert(tk.END, "[Image content copied]")
        else:
            # Text content
            self.image_preview.pack_forget()
            self.clipboard_preview.pack(fill=tk.X)
            self.clipboard_preview.delete("1.0", tk.END)
            self.clipboard_preview.insert(tk.END, content)
            
            if self.auto_send.get():
                # Automatically send if auto-send is enabled
                self.send_message()
            else:
                # Show preview when auto-send is disabled
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
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # API Configuration Section
        ttk.Label(frame, text="API Configuration", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, columnspan=4, pady=(0, 20), sticky="w")
        
        ttk.Label(frame, text="Provider", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=0, pady=(0, 10), sticky="w")
        ttk.Label(frame, text="API Key", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=1, pady=(0, 10), sticky="w", padx=(20, 0))
        ttk.Label(frame, text="Model", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=2, pady=(0, 10), sticky="w", padx=(20, 0))
        ttk.Label(frame, text="Primary", font=("Segoe UI", 10, "bold")).grid(
            row=1, column=3, pady=(0, 10), sticky="w", padx=(20, 0))
        
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
                "system_prompt": system_prompt.get("1.0", tk.END).strip()
            }
            with open(get_config_path(), 'w') as f:
                json.dump(config_data, f, indent=4)
            active_key = config_data["providers"][active]["api_key"]
            active_model = config_data["providers"][active]["model"]
            self.llm_client = LLMClient(active_key, provider=active, model=active_model)
            dialog.destroy()
            self.status.config(text="Config saved!")
            self.start_monitoring()
        
        btn_save = ttk.Button(frame, text="Save", command=save)
        btn_save.grid(row=7, column=0, columnspan=4, pady=(20, 0))
        
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

    def show_readme(self):
        readme_dialog = tk.Toplevel(self.root)
        readme_dialog.title("Readme")
        readme_dialog.geometry("500x400")
        readme_dialog.transient(self.root)
        readme_dialog.grab_set()
        
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