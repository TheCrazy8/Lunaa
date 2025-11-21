
import ollama
from ollama import chat
from ollama import ChatResponse
import tkinter as tk
from tkinter import ttk
import threading
import torch
try:
    import sv_ttk
except ImportError:
    sv_ttk = None  # Themeing is optional

import os, io, base64, datetime, json, re, subprocess, time, sys
try:
    from PIL import Image, ImageTk
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False

# HTTP dependency (used for /web and Automatic1111)
try:
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:
    _REQUESTS_AVAILABLE = False

# Import new Lunaa modules
try:
    from lunaa_modules.memory.memory_engine import MemoryEngine
    from lunaa_modules.vision.vision_model import VisionModel
    from lunaa_modules.audio.audio_processor import AudioProcessor
    from lunaa_modules.data_sources.data_manager import DataSourceManager
    from lunaa_modules.context.context_engine import ContextEngine
    from lunaa_modules.command_api.api_server import CommandAPI
    from lunaa_modules.tools.math_engine import MathEngine
    from lunaa_modules.tools.geolocation import GeolocationEngine
    from lunaa_modules.tools.file_viewer import FileViewer
    from lunaa_modules.tools.web_scraper import WebScraper
    from lunaa_modules.extensions.extension_manager import ExtensionManager
    _MODULES_AVAILABLE = True
except ImportError as e:
    _MODULES_AVAILABLE = False
    print(f"Warning: Some modules not available: {e}")

# Fixed text model (Ollama)
TEXT_MODEL = 'gemma3'

###############################################
# Image Backend: Automatic1111 WebUI
###############################################
# Reverting to Automatic1111 as requested. Ensure the WebUI is running locally.
IMG_BACKEND = 'auto1111'
SD_API_URL = os.environ.get('SD_API_URL', 'http://127.0.0.1:7860')
SD_TXT2IMG = f'{SD_API_URL}/sdapi/v1/txt2img'
SD_PROGRESS = f'{SD_API_URL}/sdapi/v1/progress'
SD_STATUS = {'online': False, 'last_error': None}

# Keep track of which models we've already pulled in this session (for Ollama text only)
loaded_models = set()
_image_refs = []  # Prevent GC of PhotoImages

def check_sd(timeout=4):
    if not _REQUESTS_AVAILABLE:
        SD_STATUS['online'] = False
        SD_STATUS['last_error'] = 'requests not installed'
        return False
    try:
        r = requests.get(SD_PROGRESS, timeout=timeout)
        if r.status_code == 200:
            SD_STATUS['online'] = True
            SD_STATUS['last_error'] = None
            return True
        SD_STATUS['online'] = False
        SD_STATUS['last_error'] = f'HTTP {r.status_code}'
        return False
    except Exception as e:
        SD_STATUS['online'] = False
        SD_STATUS['last_error'] = str(e)
        return False


def main():
    """Tkinter chat UI with enhanced features: memory, vision, audio, data sources, and more."""
    root = tk.Tk()
    root.title("Lunaa AI - Enhanced Edition")

    # Initialize enhanced modules
    if _MODULES_AVAILABLE:
        memory_engine = MemoryEngine()
        vision_model = VisionModel()
        audio_processor = AudioProcessor()
        data_manager = DataSourceManager()
        context_engine = ContextEngine()
        command_api = CommandAPI()
        math_engine = MathEngine()
        geo_engine = GeolocationEngine()
        file_viewer = FileViewer()
        web_scraper = WebScraper()
        extension_manager = ExtensionManager()
    else:
        memory_engine = None
        vision_model = None
        audio_processor = None
        data_manager = None
        context_engine = None
        command_api = None
        math_engine = None
        geo_engine = None
        file_viewer = None
        web_scraper = None
        extension_manager = None

    history_lock = threading.Lock()
    messages = [{'role': 'system', 'content': "You are Lunaa, a helpful AI assistant with enhanced capabilities including memory, vision, file viewing, mathematics, geolocation, and more. You may be given fetched web excerpts labeled 'WEB EXCERPT' to ground answers."}]

    # --- Top bar ---
    top_frame = ttk.Frame(root, padding=(10, 10, 10, 0))
    top_frame.grid(row=0, column=0, sticky='ew')
    root.columnconfigure(0, weight=1)

    ttk.Label(top_frame, text=f"Text model: {TEXT_MODEL}").grid(row=0, column=0, sticky='w')
    ttk.Label(top_frame, text=f"Image: Automatic1111 WebUI").grid(row=0, column=1, padx=15, sticky='w')

    pull_status_var = tk.StringVar(value="Initializing…")
    ttk.Label(top_frame, textvariable=pull_status_var, foreground='#888').grid(row=0, column=2, padx=10, sticky='w')

    # Conversation area
    text_frame = ttk.Frame(root, padding=10)
    text_frame.grid(row=1, column=0, sticky='nsew')
    root.rowconfigure(1, weight=1)
    convo = tk.Text(text_frame, wrap='word', height=20, width=80, state='disabled')
    convo.grid(row=0, column=0, columnspan=6, sticky='nsew')
    scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=convo.yview)
    scrollbar.grid(row=0, column=6, sticky='ns')
    convo.configure(yscrollcommand=scrollbar.set)
    text_frame.rowconfigure(0, weight=1)
    text_frame.columnconfigure(0, weight=1)

    entry_var = tk.StringVar()
    entry = ttk.Entry(text_frame, textvariable=entry_var, width=60)
    entry.grid(row=1, column=0, columnspan=2, sticky='ew', pady=(8,0))
    text_frame.columnconfigure(0, weight=1)

    send_btn = ttk.Button(text_frame, text='Send')
    send_btn.grid(row=1, column=2, padx=5, pady=(8,0), sticky='e')
    clear_btn = ttk.Button(text_frame, text='Clear')
    clear_btn.grid(row=1, column=3, padx=5, pady=(8,0))
    dump_btn = ttk.Button(text_frame, text='Dump History')
    dump_btn.grid(row=1, column=4, padx=5, pady=(8,0))
    quit_btn = ttk.Button(text_frame, text='Quit', command=root.destroy)
    quit_btn.grid(row=1, column=5, padx=5, pady=(8,0))

    streaming_active = {'value': False}
    streaming_buffer = {'text': ''}

    # --- Utility append functions ---
    def append(text: str):
        convo.configure(state='normal')
        convo.insert('end', text + '\n')
        convo.see('end')
        convo.configure(state='disabled')
        print(text)

    def append_no_newline(text: str):
        convo.configure(state='normal')
        convo.insert('end', text)
        convo.see('end')
        convo.configure(state='disabled')

    def embed_image(pil_image, caption: str):
        if not _PIL_AVAILABLE:
            append(f"[PIL not installed: {caption}]")
            return
        photo = ImageTk.PhotoImage(pil_image)
        _image_refs.append(photo)
        convo.configure(state='normal')
        convo.insert('end', caption + '\n')
        convo.image_create('end', image=photo)
        convo.insert('end', '\n')
        convo.see('end')
        convo.configure(state='disabled')

    def format_history_lines(include_system=False):
        with history_lock:
            for msg in messages:
                if msg['role'] == 'system' and not include_system:
                    continue
                yield f"[{msg['role']}] {msg['content']}"

    def print_history(include_system=False):
        print('\n=== Conversation History ===')
        for line in format_history_lines(include_system=include_system):
            print(line)
        print('=== End History ===\n')

    def clear_conversation():
        with history_lock:
            system = messages[0]
            messages.clear()
            messages.append(system)
        convo.configure(state='normal')
        convo.delete('1.0','end')
        convo.configure(state='disabled')
        append('[Conversation cleared]')
        print_history(include_system=True)

    clear_btn.configure(command=clear_conversation)

    # --- Model pull (text only) ---
    def pull_model_if_needed(model_name: str):
        if model_name in loaded_models:
            return
        try:
            pull_status_var.set(f'Pulling {model_name}…')
            ollama.pull(model_name)
            loaded_models.add(model_name)
            pull_status_var.set(f'{model_name} ready')
        except Exception as e:
            pull_status_var.set(f'Pull failed {model_name}')
            append(f"[Error pulling {model_name}: {e}]")

    # --- Initial startup: pull text model + check A1111 ---
    def initial_startup():
        pull_model_if_needed(TEXT_MODEL)
        online = check_sd()
        if online:
            pull_status_var.set('Text ready / A1111 online')
        else:
            pull_status_var.set(f'Text ready / A1111 offline ({SD_STATUS["last_error"]})')
    threading.Thread(target=initial_startup, daemon=True).start()

    # --- Streaming helpers ---
    def begin_stream():
        streaming_active['value'] = True
        streaming_buffer['text'] = ''
        convo.configure(state='normal')
        convo.insert('end', f'{TEXT_MODEL}: ')
        convo.see('end')
        convo.configure(state='disabled')

    def stream_token(delta: str):
        if not streaming_active['value']:
            return
        streaming_buffer['text'] += delta
        append_no_newline(delta)

    def end_stream():
        if not streaming_active['value']:
            return
        convo.configure(state='normal')
        convo.insert('end', '\n')
        convo.configure(state='disabled')
        streaming_active['value'] = False

    # --- /img parsing ---
    def parse_image_command(raw: str):
        body = raw[len('/img'):].strip()
        if '||' in body:
            prompt, negative = body.split('||', 1)
            return prompt.strip(), negative.strip()
        return body.strip(), ''

    # --- Automatic1111 generation ---
    def generate_image(prompt: str, negative: str):
        if IMG_BACKEND != 'auto1111':
            append(f'[Unsupported backend {IMG_BACKEND}]')
            return
        if not _REQUESTS_AVAILABLE:
            append('[requests not installed: cannot reach A1111 API]')
            return
        if not SD_STATUS['online']:
            check_sd()
        if not SD_STATUS['online']:
            append(f'[A1111 offline: {SD_STATUS["last_error"]}]')
            return
        payload = {
            'prompt': prompt,
            'negative_prompt': negative,
            'steps': 30,
            'width': 512,
            'height': 512,
            'sampler_name': 'Euler a'
        }
        try:
            resp = requests.post(SD_TXT2IMG, json=payload, timeout=180)
        except Exception as e:
            SD_STATUS['online'] = False
            SD_STATUS['last_error'] = str(e)
            append(f'[A1111 request error: {e}]')
            return
        if resp.status_code != 200:
            append(f'[A1111 HTTP {resp.status_code}: {resp.text[:120]}]')
            return
        try:
            data = resp.json()
        except Exception as e:
            append(f'[A1111 JSON error: {e}]')
            return
        images = data.get('images') or []
        if not images:
            append('[A1111 returned no images]')
            return
        os.makedirs('generated_images', exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        for idx, b64img in enumerate(images):
            try:
                binary = base64.b64decode(b64img)
            except Exception as e:
                append(f'[Decode failed: {e}]')
                continue
            filename = f'generated_images/{ts}_{idx}.png'
            try:
                with open(filename, 'wb') as f:
                    f.write(binary)
            except Exception as e:
                append(f'[Write failed {filename}: {e}]')
                continue
            if _PIL_AVAILABLE:
                try:
                    img = Image.open(io.BytesIO(binary))
                except Exception as e:
                    append(f'[PIL open failed: {e}]')
                    continue
                embed_image(img, f'Image saved: {filename}')
            else:
                append(f'Image saved: {filename} (install Pillow to preview)')

    # --- /web helpers (unchanged logic) ---
    def sanitize_url_or_query(arg: str):
        if re.match(r'^https?://', arg, re.I):
            return arg, None
        return None, arg

    def duckduckgo_search(query: str, max_results=3):
        if not _REQUESTS_AVAILABLE:
            return []
        try:
            r = requests.get('https://duckduckgo.com/html/', params={'q': query}, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code != 200:
                return []
            links = re.findall(r'<a rel=\"nofollow\" class=\"result__a\" href=\"(.*?)\"', r.text)
            cleaned = []
            for l in links:
                if l.startswith('http') and 'duckduckgo.com' not in l:
                    cleaned.append(l)
                if len(cleaned) >= max_results:
                    break
            return cleaned
        except Exception:
            return []

    def fetch_url(url: str, max_chars=6000):
        if not _REQUESTS_AVAILABLE:
            return '[requests not installed]', ''
        try:
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code != 200:
                return f'[HTTP {resp.status_code}]', ''
            text = resp.text
        except Exception as e:
            return f'[Fetch error: {e}]', ''
        stripped = re.sub(r'<script.*?</script>|<style.*?</style>', '', text, flags=re.S|re.I)
        stripped = re.sub(r'<[^>]+>', ' ', stripped)
        stripped = re.sub(r'&nbsp;|&amp;|&lt;|&gt;', ' ', stripped)
        stripped = re.sub(r'\s+', ' ', stripped).strip()
        snippet = stripped[:max_chars]
        return None, snippet

    def summarize_snippet(snippet: str):
        if not snippet:
            return '[Empty snippet]'
        chunks = []
        size = 1000
        for i in range(0, len(snippet), size):
            chunks.append(snippet[i:i+size])
        summaries = []
        for idx, ch in enumerate(chunks[:4]):
            try:
                resp = chat(model=TEXT_MODEL, messages=[
                    {'role': 'system', 'content': 'Summarize the provided web text accurately and concisely.'},
                    {'role': 'user', 'content': ch}
                ])
                summaries.append(resp['message']['content'])
            except Exception as e:
                summaries.append(f'[Summarization error chunk {idx}: {e}]')
        return '\n'.join(summaries)

    def handle_web_command(raw: str):
        arg = raw[len('/web'):].strip()
        if not arg:
            append('[Usage] /web <url or search terms>')
            finalize_web(None)
            return
        url, query = sanitize_url_or_query(arg)
        urls = []
        if url:
            urls = [url]
        else:
            append(f'[Searching: {query}]')
            urls = duckduckgo_search(query, max_results=2)
            if not urls:
                append('[No search results]')
                finalize_web(None)
                return
        append(f'[Fetching {len(urls)} source(s)]')
        def worker():
            combined_contexts = []
            for u in urls:
                err, snippet = fetch_url(u)
                if err:
                    root.after(0, lambda e=err: append(f'{u} -> {e}'))
                    continue
                summary = summarize_snippet(snippet)
                combined_contexts.append(f'URL: {u}\nSUMMARY:\n{summary}')
            if not combined_contexts:
                root.after(0, lambda: finalize_web(None))
                return
            web_block = '\n\n'.join(combined_contexts)
            with history_lock:
                messages.append({'role': 'system', 'content': f'WEB EXCERPT BEGIN\n{web_block}\nWEB EXCERPT END'})
            root.after(0, lambda: finalize_web(len(combined_contexts)))
        threading.Thread(target=worker, daemon=True).start()

    def finalize_web(ct):
        send_btn.configure(state='normal')
        entry.configure(state='normal')
        entry_var.set('')
        if ct:
            append(f'[Added {ct} summarized source(s) to context]')
        entry.focus()

    # --- Text inference ---
    def do_inference(user_msg: str):
        pull_model_if_needed(TEXT_MODEL)
        try:
            with history_lock:
                temp_history = messages + [{'role': 'user', 'content': user_msg}]
            print(f'--- Streaming response from {TEXT_MODEL} ---')
            root.after(0, begin_stream)
            full_chunks = []
            stream_resp = chat(model=TEXT_MODEL, messages=temp_history, stream=True)
            for part in stream_resp:
                delta = part.get('message', {}).get('content', '')
                if delta:
                    full_chunks.append(delta)
                    root.after(0, lambda d=delta: stream_token(d))
                    print(delta, end='', flush=True)
            print('\n--- End of streamed response ---')
            resp_text = ''.join(full_chunks).strip() or '[No content returned]'
        except Exception as e:
            resp_text = f'[Error querying model {TEXT_MODEL}: {e}]'
        root.after(0, lambda: finalize_response(user_msg, resp_text))

    def finalize_response(user_msg: str, resp_text: str):
        end_stream()
        with history_lock:
            messages.append({'role': 'user', 'content': user_msg})
            messages.append({'role': 'assistant', 'content': resp_text})
        
        # Add to memory and context if available
        if _MODULES_AVAILABLE:
            if memory_engine:
                memory_engine.add_conversation('assistant', resp_text)
            if context_engine:
                context_engine.add_to_context({'role': 'assistant', 'content': resp_text})
        
        print_history(include_system=False)
        send_btn.configure(state='normal')
        entry.configure(state='normal')
        entry_var.set('')
        entry.focus()

    # --- Handlers ---
    def handle_image_command(raw: str):
        prompt, negative = parse_image_command(raw)
        if not prompt:
            append('[Usage] /img <prompt> [|| negative prompt]')
            send_btn.configure(state='normal')
            entry.configure(state='normal')
            return
        append(f'You (image prompt): {prompt}' + (f' | neg: {negative}' if negative else ''))
        with history_lock:
            summary = prompt if len(prompt) < 120 else prompt[:117] + '...'
            messages.append({'role': 'user', 'content': f'(image prompt) {summary}'})
        threading.Thread(target=generate_image, args=(prompt, negative), daemon=True).start()
        send_btn.configure(state='normal')
        entry.configure(state='normal')
        entry_var.set('')
        entry.focus()

    def handle_sdstatus():
        online = check_sd()
        append('[A1111 online]' if online else f'[A1111 offline: {SD_STATUS["last_error"]}]')

    # --- New command handlers for enhanced features ---
    def handle_vision_command(raw: str):
        """Handle /vision <image_path> [question]"""
        if not _MODULES_AVAILABLE or vision_model is None:
            append('[Vision module not available]')
            reset_input_state()
            return
        
        parts = raw[len('/vision'):].strip().split(maxsplit=1)
        if not parts:
            append('[Usage] /vision <image_path> [question]')
            reset_input_state()
            return
        
        image_path = parts[0]
        question = parts[1] if len(parts) > 1 else None
        
        def worker():
            if question:
                result = vision_model.analyze_image_with_question(image_path, question)
            else:
                result = vision_model.analyze_image(image_path)
            root.after(0, lambda: append(f'[Vision] {result}'))
            root.after(0, reset_input_state)
        
        threading.Thread(target=worker, daemon=True).start()
    
    def handle_math_command(raw: str):
        """Handle /math <expression> or /plot <expression>"""
        if not _MODULES_AVAILABLE or math_engine is None:
            append('[Math module not available]')
            reset_input_state()
            return
        
        if raw.startswith('/plot'):
            expr = raw[len('/plot'):].strip()
            if not expr:
                append('[Usage] /plot <expression>')
                reset_input_state()
                return
            result = math_engine.plot_function(expr)
            append(f'[Math] {result}')
        else:
            expr = raw[len('/math'):].strip()
            if not expr:
                append('[Usage] /math <expression>')
                reset_input_state()
                return
            result = math_engine.calculate(expr)
            append(f'[Math] {result}')
        reset_input_state()
    
    def handle_file_command(raw: str):
        """Handle /file <path> or /dir <path>"""
        if not _MODULES_AVAILABLE or file_viewer is None:
            append('[File viewer module not available]')
            reset_input_state()
            return
        
        if raw.startswith('/dir'):
            path = raw[len('/dir'):].strip() or '.'
            result = file_viewer.list_directory(path)
        else:
            path = raw[len('/file'):].strip()
            if not path:
                append('[Usage] /file <filepath>')
                reset_input_state()
                return
            result = file_viewer.view_file(path)
        
        append(f'[File]\n{result}')
        reset_input_state()
    
    def handle_memory_command(raw: str):
        """Handle /memory commands"""
        if not _MODULES_AVAILABLE or memory_engine is None:
            append('[Memory module not available]')
            reset_input_state()
            return
        
        parts = raw[len('/memory'):].strip().split(maxsplit=1)
        if not parts or parts[0] == 'help':
            append('[Memory Commands] /memory add <fact> | /memory search <query> | /memory clear')
            reset_input_state()
            return
        
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ''
        
        if cmd == 'add' and arg:
            memory_engine.add_fact(arg)
            append(f'[Memory] Fact added: {arg}')
        elif cmd == 'search' and arg:
            results = memory_engine.search_facts(arg)
            if results:
                append(f'[Memory] Found {len(results)} facts:')
                for fact in results[:5]:
                    append(f"  - {fact['fact']}")
            else:
                append('[Memory] No matching facts found')
        elif cmd == 'clear':
            memory_engine.clear_memory()
            append('[Memory] Memory cleared')
        else:
            append('[Memory] Unknown command. Use /memory help')
        reset_input_state()
    
    def handle_dataset_command(raw: str):
        """Handle /dataset commands"""
        if not _MODULES_AVAILABLE or data_manager is None:
            append('[Data source module not available]')
            reset_input_state()
            return
        
        parts = raw[len('/dataset'):].strip().split(maxsplit=1)
        if not parts:
            append('[Usage] /dataset load <name> | /dataset query <name>')
            reset_input_state()
            return
        
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ''
        
        if cmd == 'load' and arg:
            def worker():
                result = data_manager.load_huggingface_dataset(arg)
                root.after(0, lambda: append(f'[Dataset] {result}'))
                root.after(0, reset_input_state)
            threading.Thread(target=worker, daemon=True).start()
        elif cmd == 'query' and arg:
            result = data_manager.query_dataset(arg)
            append(f'[Dataset]\n{result}')
            reset_input_state()
        else:
            append('[Dataset] Unknown command')
            reset_input_state()
    
    def handle_geo_command(raw: str):
        """Handle /geo commands"""
        if not _MODULES_AVAILABLE or geo_engine is None:
            append('[Geolocation module not available]')
            reset_input_state()
            return
        
        arg = raw[len('/geo'):].strip()
        if not arg:
            append('[Usage] /geo <address>')
            reset_input_state()
            return
        
        result = geo_engine.geocode(arg)
        append(f'[Geo] {result}')
        reset_input_state()
    
    def handle_extension_command(raw: str):
        """Handle /ext commands"""
        if not _MODULES_AVAILABLE or extension_manager is None:
            append('[Extension module not available]')
            reset_input_state()
            return
        
        parts = raw[len('/ext'):].strip().split(maxsplit=1)
        if not parts:
            append('[Extensions] /ext load <name> | /ext unload <name> | /ext list')
            reset_input_state()
            return
        
        cmd = parts[0]
        arg = parts[1] if len(parts) > 1 else ''
        
        if cmd == 'load' and arg:
            result = extension_manager.load_extension(arg)
            append(f'[Extension] {result}')
        elif cmd == 'unload' and arg:
            result = extension_manager.unload_extension(arg)
            append(f'[Extension] {result}')
        elif cmd == 'list':
            result = extension_manager.list_extensions()
            append(f'[Extensions]\n{result}')
        else:
            append('[Extension] Unknown command')
        reset_input_state()
    
    def handle_context_command(raw: str):
        """Handle /context command"""
        if not _MODULES_AVAILABLE or context_engine is None:
            append('[Context module not available]')
            reset_input_state()
            return
        
        summary = context_engine.get_context_summary()
        append(f'[Context] {summary}')
        reset_input_state()

    def handle_help_command():
        """Display help information"""
        append('=== Lunaa AI Commands ===')
        append('Image: /img <prompt> [|| negative]')
        append('Vision: /vision <image_path> [question]')
        append('Web: /web <url or search>')
        append('Math: /math <expression> or /plot <expression>')
        append('Files: /file <path> or /dir [path]')
        append('Memory: /memory add/search/clear <text>')
        append('Datasets: /dataset load/query <name>')
        append('Location: /geo <address>')
        append('Extensions: /ext load/unload/list <name>')
        append('Context: /context')
        append('Status: /sdstatus')
        append('See COMMANDS.md for detailed documentation')
        reset_input_state()

    def send(event=None):
        user_msg = entry_var.get().strip()
        if not user_msg:
            return
        send_btn.configure(state='disabled')
        entry.configure(state='disabled')
        if user_msg.startswith('/help'):
            handle_help_command()
            return
        if user_msg.startswith('/img'):
            handle_image_command(user_msg)
            return
        if user_msg.startswith('/web'):
            handle_web_command(user_msg)
            return
        if user_msg.startswith('/sdstatus'):
            handle_sdstatus()
            reset_input_state()
            return
        if user_msg.startswith('/vision'):
            handle_vision_command(user_msg)
            return
        if user_msg.startswith('/math'):
            handle_math_command(user_msg)
            return
        if user_msg.startswith('/plot'):
            handle_math_command(user_msg)
            return
        if user_msg.startswith('/file'):
            handle_file_command(user_msg)
            return
        if user_msg.startswith('/dir'):
            handle_file_command(user_msg)
            return
        if user_msg.startswith('/memory'):
            handle_memory_command(user_msg)
            return
        if user_msg.startswith('/dataset'):
            handle_dataset_command(user_msg)
            return
        if user_msg.startswith('/geo'):
            handle_geo_command(user_msg)
            return
        if user_msg.startswith('/ext'):
            handle_extension_command(user_msg)
            return
        if user_msg.startswith('/context'):
            handle_context_command(user_msg)
            return
        append(f'You: {user_msg}')
        
        # Add to memory and context if available
        if _MODULES_AVAILABLE:
            if memory_engine:
                memory_engine.add_conversation('user', user_msg)
            if context_engine:
                context_engine.add_to_context({'role': 'user', 'content': user_msg})
        
        threading.Thread(target=do_inference, args=(user_msg,), daemon=True).start()

    def reset_input_state():
        send_btn.configure(state='normal')
        entry.configure(state='normal')
        entry_var.set('')
        entry.focus()

    def dump_history_action():
        print_history(include_system=True)

    dump_btn.configure(command=dump_history_action)
    send_btn.configure(command=send)
    entry.bind('<Return>', send)

    # Initial message
    if not _REQUESTS_AVAILABLE:
        append('Install requests (pip install requests) for /web & /img & SD autostart.')
    
    append('=== Lunaa AI - Enhanced Edition ===')
    append('Type /help to see all available commands')
    append('')
    append('Quick commands:')
    append('  /img <prompt> [|| negative] - Generate image')
    append('  /web <url or search> - Web search')
    append('  /vision <image_path> - Analyze image')
    append('  /math <expression> - Calculate')
    append('  /file <path> - View file')
    append('  /memory add/search - Memory operations')
    
    if not _REQUESTS_AVAILABLE:
        append('')
        append('Install requests: pip install requests')
    
    entry.focus()

    if sv_ttk is not None:
        try:
            sv_ttk.set_theme('dark')
        except Exception:
            pass

    root.mainloop()

if __name__ == '__main__':
    main()
