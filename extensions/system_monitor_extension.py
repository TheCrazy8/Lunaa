import platform
import os
import shutil
import time

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

def initialize():
    """
    /sys cpu
    /sys mem
    /sys gpu
    /sys summary
    """
    return {"sys": handle_sys_command}

def handle_sys_command(raw: str, append, ask_llm=None):
    sub = (raw.strip().split() or ["summary"])[0].lower()
    if sub == "cpu":
        _show_cpu(append)
    elif sub == "mem":
        _show_mem(append)
    elif sub == "gpu":
        _show_gpu(append)
    elif sub == "summary":
        _show_summary(append)
    else:
        append("=== /sys commands ===")
        append("/sys summary   - basic system overview")
        append("/sys cpu       - CPU info and usage")
        append("/sys mem       - memory usage")
        append("/sys gpu       - CUDA/GPU info (if available)")

def _show_cpu(append):
    append("=== CPU Info ===")
    append(f"Platform: {platform.platform()}")
    append(f"Processor: {platform.processor() or 'N/A'}")
    if _PSUTIL_OK:
        try:
            count_logical = psutil.cpu_count(logical=True)
            count_physical = psutil.cpu_count(logical=False)
            append(f"Cores: {count_physical} physical, {count_logical} logical")
            percent = psutil.cpu_percent(interval=0.5)
            append(f"CPU usage: {percent}%")
        except Exception as e:
            append(f"[sys] psutil error: {e}")
    else:
        append("psutil not installed; install with 'pip install psutil' for CPU usage details.")

def _show_mem(append):
    append("=== Memory Info ===")
    if _PSUTIL_OK:
        try:
            m = psutil.virtual_memory()
            total_gb = m.total / (1024**3)
            used_gb = (m.total - m.available) / (1024**3)
            append(f"RAM: {used_gb:.2f} / {total_gb:.2f} GB used ({m.percent}%)")
        except Exception as e:
            append(f"[sys] psutil error: {e}")
    else:
        append("psutil not installed; install with 'pip install psutil' for detailed memory info.")

    # Disk
    try:
        usage = shutil.disk_usage(os.getcwd())
        total = usage.total / (1024**3)
        used = (usage.total - usage.free) / (1024**3)
        append(f"Disk (cwd): {used:.2f} / {total:.2f} GB used")
    except Exception as e:
        append(f"[sys] Disk usage error: {e}")

def _show_gpu(append):
    append("=== GPU Info ===")
    if not _TORCH_OK:
        append("torch not installed; cannot inspect CUDA/GPU.")
        return
    try:
        if not torch.cuda.is_available():
            append("CUDA not available.")
            return
        count = torch.cuda.device_count()
        append(f"CUDA devices: {count}")
        for idx in range(count):
            name = torch.cuda.get_device_name(idx)
            cap = torch.cuda.get_device_capability(idx)
            mem_total = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
            append(f"GPU {idx}: {name}, capability {cap[0]}.{cap[1]}, mem {mem_total:.2f} GB")
    except Exception as e:
        append(f"[sys] torch.cuda error: {e}")

def _show_summary(append):
    append("=== System Summary ===")
    append(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    append(f"Python: {platform.python_version()}")
    _show_cpu(append)
    _show_mem(append)
    _show_gpu(append)