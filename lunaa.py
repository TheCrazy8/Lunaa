import os
import threading
import tkinter as tk
import tkinter.ttk as ttk

# Optional theme; install with: pip install sv-ttk
try:
    import sv_ttk
except Exception:
    sv_ttk = None

# Ollama Python SDK; install with: pip install ollama
# Make sure the Ollama daemon is running (ollama serve) and a model is pulled, e.g.:
#   ollama pull llama3.1
import ollama


DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful, concise assistant. Answer clearly and directly."
)


class OllamaChat:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = SYSTEM_PROMPT):
        self.model = model
        self.history = []
        if system_prompt:
            self.history.append({"role": "system", "content": system_prompt})

    def ask_stream(self, user_text: str):
        """
        Generator that yields chunks of assistant text as they stream from Ollama.
        Also updates the internal conversation history when the response completes.
        """
        self.history.append({"role": "user", "content": user_text})
        accumulated = []

        try:
            # stream=True yields dicts with "message": {"role": "assistant", "content": "<chunk>"}
            for chunk in ollama.chat(model=self.model, messages=self.history, stream=True):
                delta = chunk.get("message", {}).get("content", "")
                if delta:
                    accumulated.append(delta)
                    yield delta
        except Exception as e:
            # Surface the error as a single yielded message
            err = f"[Error] {e}\nMake sure the Ollama daemon is running and the model '{self.model}' is pulled."
            yield err
            # Roll back the last user message on error to keep history consistent
            self.history.pop()
            return

        # Save the completed assistant message into history
        full = "".join(accumulated)
        self.history.append({"role": "assistant", "content": full})


class ChatUI(tk.Tk):
    def __init__(self, model=DEFAULT_MODEL):
        super().__init__()
        self.title(f"Lunaa • Ollama Chatbot ({model})")
        self.geometry("820x600")

        if sv_ttk:
            try:
                sv_ttk.set_theme("dark")
            except Exception:
                pass

        self.chat = OllamaChat(model=model)

        # Layout: top frame for transcript, bottom for input
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        transcript_frame = ttk.Frame(self)
        transcript_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        transcript_frame.rowconfigure(0, weight=1)
        transcript_frame.columnconfigure(0, weight=1)

        self.text = tk.Text(
            transcript_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 11),
            padx=8,
            pady=8
        )
        self.text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(transcript_frame, command=self.text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=scrollbar.set)

        # Tags for styling
        self.text.tag_configure("user", foreground="#0b5fff", font=("Segoe UI", 11, "bold"))
        self.text.tag_configure("assistant", foreground="#10a37f")
        self.text.tag_configure("meta", foreground="#888888", font=("Segoe UI", 9, "italic"))

        input_frame = ttk.Frame(self)
        input_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))
        input_frame.columnconfigure(0, weight=1)

        self.entry = ttk.Entry(input_frame)
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.entry.bind("<Return>", self.on_send)

        self.send_btn = ttk.Button(input_frame, text="Send", command=self.on_send)
        self.send_btn.grid(row=0, column=1)

        # Optionally show model info
        self._append_meta(f"Using model: {model}\nType your message and press Enter.\n")

    def _append(self, text: str, tag=None):
        self.text.configure(state="normal")
        if tag:
            self.text.insert("end", text, tag)
        else:
            self.text.insert("end", text)
        self.text.insert("end", "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def _append_meta(self, text: str):
        self._append(text, "meta")

    def _append_user(self, text: str):
        self._append(f"You: {text}", "user")

    def _append_assistant_header(self):
        # Insert the assistant header and return index where the body starts
        self.text.configure(state="normal")
        index_start = self.text.index("end-1c")
        self.text.insert("end", "Assistant: ", "assistant")
        self.text.configure(state="disabled")
        self.text.see("end")
        return index_start

    def _append_assistant_chunk(self, chunk: str):
        self.text.configure(state="normal")
        self.text.insert("end", chunk, "assistant")
        self.text.see("end")
        self.text.configure(state="disabled")

    def set_input_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
        self.send_btn.configure(state=state)

    def on_send(self, event=None):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._append_user(text)
        self.set_input_enabled(False)

        # Start streaming response in a background thread
        threading.Thread(target=self._stream_response, args=(text,), daemon=True).start()

    def _stream_response(self, user_text: str):
        # Ensure the assistant header is on the UI thread
        self.after(0, self._append_assistant_header)

        def do_stream():
            for chunk in self.chat.ask_stream(user_text):
                self.after(0, self._append_assistant_chunk, chunk)
            self.after(0, self.set_input_enabled, True)

        # Run the actual streaming loop
        do_stream()


if __name__ == "__main__":
    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    app = ChatUI(model=model)
    app.mainloop()