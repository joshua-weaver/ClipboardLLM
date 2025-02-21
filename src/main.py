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
import sv_ttk  # Use sv_ttk for Sun Valley theme

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
        if self.provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.model = model or "gpt-4"
        elif self.provider == "anthropic":
            self.endpoint = "https://api.anthropic.com/v1/messages"
            self.model = model or "claude-3-sonnet-20240229"
        elif self.provider == "gemini":
            self.endpoint = None
            self.model = model or "models/gemini-2.0-flash"
        else:
            self.endpoint = ""

    def process_content(self, content: str) -> str:
        try:
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
                    response = model.generate_content(content)
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
                    "messages": [{"role": "user", "content": content}],
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

    def start(self):
        self.running = True
        threading.Thread(target=self._monitor, daemon=True).start()

    def stop(self):
        self.running = False

    def _monitor(self):
        while self.running:
            try:
                win32clipboard.OpenClipboard()
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                    content = win32clipboard.GetClipboardData(win32con.CF_TEXT).decode('utf-8')
                    if content != self.previous_content:
                        self.previous_content = content
                        self.callback(content)
                win32clipboard.CloseClipboard()
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
        input_frame.columnconfigure(0, weight=1)

        self.status = ttk.Label(frame, text="Starting...")
        self.status.pack(fill=tk.X, pady=(5, 0))

        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure API Keys", command=self.show_settings)
        settings_menu.add_command(label="Readme", command=self.show_readme)

        self.monitor = ClipboardMonitor(self.on_clipboard_change)
        self.llm_client = None
        self.check_config()

    def on_enter(self, event):
        if not event.state & 0x1:  # Shift key not pressed
            self.send_message()
            return "break"
        return None

    def send_message(self):
        message = self.input_text.get("1.0", tk.END).strip()
        if not message:
            return
        if not self.llm_client:
            self.add_message("Error: Please configure API key first", is_user=False)
            return
        self.input_text.delete("1.0", tk.END)
        self.add_message(message, is_user=True)
        self.status.config(text="Processing message...")
        def process():
            response = self.llm_client.process_content(message)
            self.root.after(0, lambda: self.add_message(response, is_user=False))
            self.root.after(0, lambda: self.status.config(text="Ready"))
        threading.Thread(target=process, daemon=True).start()

    def on_clipboard_change(self, content):
        if not self.llm_client:
            self.add_message("Error: Please configure API key first", is_user=False)
            return
        self.add_message(f"Copied: {content}", is_user=True)
        self.status.config(text="Processing clipboard content...")
        def process():
            response = self.llm_client.process_content(content)
            self.root.after(0, lambda: self.add_message(response, is_user=False))
            self.root.after(0, lambda: self.status.config(text="Monitoring clipboard..."))
        threading.Thread(target=process, daemon=True).start()

    def add_message(self, message: str, is_user: bool = False):
        self.chat_text.configure(state=tk.NORMAL)
        if self.chat_text.get("1.0", tk.END).strip():
            self.chat_text.insert(tk.END, "\n\n")
        timestamp = datetime.now().strftime("%H:%M:%S")
        sender = "You" if is_user else "Assistant"
        self.chat_text.insert(tk.END, f"[{timestamp}] {sender}:\n{message}")
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
        dialog.title("API Key Configuration")
        dialog.geometry("600x500")  # Made wider to accommodate model fields
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Header labels
        ttk.Label(frame, text="Configure Providers:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=4, pady=(0, 10), sticky="w")
        
        # Column headers
        ttk.Label(frame, text="Provider", font=("Segoe UI", 9, "bold")).grid(row=1, column=0, pady=(0, 10), sticky="w")
        ttk.Label(frame, text="API Key", font=("Segoe UI", 9, "bold")).grid(row=1, column=1, pady=(0, 10), sticky="w", padx=(10, 0))
        ttk.Label(frame, text="Model", font=("Segoe UI", 9, "bold")).grid(row=1, column=2, pady=(0, 10), sticky="w")
        ttk.Label(frame, text="Primary", font=("Segoe UI", 9, "bold")).grid(row=1, column=3, pady=(0, 10), sticky="w", padx=(10, 0))
        
        # Default model values
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-sonnet-20240229",
            "gemini": "models/gemini-2.0-flash"
        }
        
        entries = {}
        model_entries = {}
        selected_provider = tk.StringVar(value="openai")
        
        for i, prov in enumerate(["openai", "anthropic", "gemini"]):
            # Provider label
            ttk.Label(frame, text=f"{prov.capitalize()}:").grid(row=i+2, column=0, sticky="w", pady=5)
            
            # API Key entry
            key_entry = ttk.Entry(frame)
            key_entry.grid(row=i+2, column=1, sticky="ew", pady=5, padx=(10, 10))
            entries[prov] = key_entry
            
            # Model entry
            model_entry = ttk.Entry(frame)
            model_entry.insert(0, default_models[prov])
            model_entry.grid(row=i+2, column=2, sticky="ew", pady=5)
            model_entries[prov] = model_entry
            
            # Radio button
            ttk.Radiobutton(
                frame,
                value=prov.lower(),
                variable=selected_provider
            ).grid(row=i+2, column=3, pady=5, padx=(10, 0))
        
        # Load existing configuration
        config_path = get_config_path()
        try:
            with open(config_path, 'r') as f:
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
        except Exception as e:
            print(f"Error loading config: {e}")

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
                }
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            active_key = config_data["providers"][active]["api_key"]
            active_model = config_data["providers"][active]["model"]
            self.llm_client = LLMClient(active_key, provider=active, model=active_model)
            dialog.destroy()
            self.status.config(text="Config saved!")
            self.start_monitoring()
        
        # Save button at the bottom
        btn_save = ttk.Button(frame, text="Save", command=save)
        btn_save.grid(row=5, column=0, columnspan=4, pady=(20, 0))
        
        # Configure grid weights
        frame.columnconfigure(1, weight=1)  # Make API Key column expandable
        frame.columnconfigure(2, weight=1)  # Make Model column expandable

    def show_readme(self):
        readme_dialog = tk.Toplevel(self.root)
        readme_dialog.title("Readme")
        readme_dialog.geometry("500x400")
        readme_dialog.transient(self.root)
        readme_dialog.grab_set()
        
        frame = ttk.Frame(readme_dialog, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        readme_text = """ClipboardLLM ClipboardLLM is an open source application licensed under the MIT License that monitors your clipboard and sends its content to various language models (OpenAI, Anthropic, Gemini) for processing. You can compile for yourself or download the latest build from the /dist/ folder.

Features
Automatic Clipboard Monitoring: Simply copy text to your clipboard, and it will be automatically sent to the selected language model. Manual Input: Enter text manually into the input box and press Enter or click Send. Easy API Configuration: Set up your API keys effortlessly via the Settings menu.

Supported Providers
OpenAI: Chat models via the OpenAI API. Anthropic: Claude models via the Anthropic API. Gemini: Models via the Google Gemini API.

Getting Started
Either compile yourself or use the .exe found in /dist/CLipboardLLM.exe. Enter the API key(s) of your choice and select your preferred provider. Then copying text will automatically send it to the configured LLM as long as ClipboardLLM is running.

License
ClipboardLLM is open source and licensed under the MIT License. https://github.com/joshua-weaver/ClipboardLLM

Support
If you'd like to support the project, send a tip via Venmo at @jshwvr or connect on X at x.com/we4v3r.

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