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
    def __init__(self, api_key, provider="openai"):
        self.api_key = api_key
        self.provider = provider.lower()
        if self.provider == "openai":
            self.endpoint = "https://api.openai.com/v1/chat/completions"
        elif self.provider == "anthropic":
            self.endpoint = "https://api.anthropic.com/v1/messages"
        elif self.provider == "gemini":
            self.endpoint = None
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
                    "model": "claude-3-sonnet-20240229",
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
                    model = genai.GenerativeModel('models/gemini-2.0-flash')
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
                    "model": "gpt-4o-mini",
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
            api_key = providers.get(active, {}).get("api_key", "")
            if not api_key:
                self.show_settings()
            else:
                self.llm_client = LLMClient(api_key, provider=active)
                self.start_monitoring()
        except Exception:
            self.show_settings()

    def show_settings(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("API Key Configuration")
        dialog.geometry("400x400")
        frame = ttk.Frame(dialog, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Provider selection with radio buttons
        selected_provider = tk.StringVar(value="openai")
        ttk.Label(frame, text="Select Primary Provider:", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        providers = ["OpenAI", "Anthropic", "Gemini"]
        for i, prov in enumerate(providers):
            ttk.Radiobutton(
                frame, 
                text=prov,
                value=prov.lower(),
                variable=selected_provider
            ).grid(row=1, column=i, padx=10, sticky="w")
        
        # API Key entries in a grid
        ttk.Label(frame, text="API Keys:", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, columnspan=3, pady=(20, 10), sticky="w")
        
        entries = {}
        for i, prov in enumerate(["openai", "anthropic", "gemini"]):
            ttk.Label(frame, text=f"{prov.capitalize()}:").grid(row=i+3, column=0, sticky="w", pady=5)
            entry = ttk.Entry(frame)
            entry.grid(row=i+3, column=1, columnspan=2, sticky="ew", pady=5, padx=(10, 0))
            entries[prov] = entry
        
        # Load existing configuration
        config_path = get_config_path()
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            active = config.get("selected_provider", "openai")
            selected_provider.set(active)
            provs = config.get("providers", {})
            entries["openai"].insert(0, provs.get("openai", {}).get("api_key", ""))
            entries["anthropic"].insert(0, provs.get("anthropic", {}).get("api_key", ""))
            entries["gemini"].insert(0, provs.get("gemini", {}).get("api_key", ""))
        except Exception as e:
            print(f"Error loading config: {e}")

        def save():
            openai_key = entries["openai"].get().strip()
            anthropic_key = entries["anthropic"].get().strip()
            gemini_key = entries["gemini"].get().strip()
            active = selected_provider.get()
            config_data = {
                "selected_provider": active,
                "providers": {
                    "openai": {"api_key": openai_key},
                    "anthropic": {"api_key": anthropic_key},
                    "gemini": {"api_key": gemini_key}
                }
            }
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            active_key = {"openai": openai_key, "anthropic": anthropic_key, "gemini": gemini_key}[active]
            self.llm_client = LLMClient(active_key, provider=active)
            dialog.destroy()
            self.status.config(text="Config saved!")
            self.start_monitoring()
        
        # Save button at the bottom
        btn_save = ttk.Button(frame, text="Save", command=save)
        btn_save.grid(row=6, column=0, columnspan=3, pady=(20, 0))
        
        # Configure grid weights
        frame.columnconfigure(1, weight=1)  # Make the entry column expandable
        frame.columnconfigure(2, weight=1)  # Make the last column expandable too

    def show_readme(self):
        readme_dialog = tk.Toplevel(self.root)
        readme_dialog.title("Readme")
        readme_dialog.geometry("500x400")
        readme_dialog.transient(self.root)
        readme_dialog.grab_set()
        
        frame = ttk.Frame(readme_dialog, padding="10")
        frame.grid(row=0, column=0, sticky="nsew")
        
        readme_text = """ClipboardLLM
ClipboardLLM is an open source application licensed under the MIT License that monitors your clipboard and sends its content to various language models (OpenAI, Anthropic, Gemini) for processing. You can compile for yourself or download the latest build from the /dist/ folder.

Features
--------
Automatic Clipboard Monitoring: Simply copy text to your clipboard, and it will be automatically sent to the selected language model.
Manual Input: Enter text manually into the input box and press Enter or click Send.
Easy API Configuration: Set up your API keys effortlessly via the Settings menu.

Supported Providers
-------------------
OpenAI: Chat models via the OpenAI API.
Anthropic: Claude models via the Anthropic API.
Gemini: Models via the Google Gemini API.

Getting Started
---------------
Enter the API key(s) of your choice and select your preferred provider. Then copying text will automatically send it to the configured LLM.

License
-------
ClipboardLLM is open source and licensed under the MIT License. https://github.com/joshua-weaver/ClipboardLLM

Support
-------
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