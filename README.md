# Lunaa AI - Enhanced Edition

A powerful AI assistant with enhanced capabilities including memory, vision, audio processing, mathematical computations, geolocation, and more.

## Features

### Core Features
- **HuggingFace Datasets Integration** - Load and query datasets from HuggingFace
- **Auto Web Search** - Fully implemented web search and scraping with DuckDuckGo
- **Vision Model** - Image analysis and visual question answering using BLIP
- **Audio Processing** - Audio recording, playback, and transcription capabilities
- **Memory Engine** - Persistent memory for conversations, facts, and context
- **Scrapy Integration** - Advanced web scraping capabilities
- **Numpy & Scipy** - Mathematical operations and scientific computing
- **File Viewing** - View and manage files directly from the interface
- **Mathematical & Graphing** - Calculate expressions and plot functions
- **Command API** - Socket-based API for external programs to interact with Lunaa
- **Context Engine** - Advanced context management for better conversations
- **Geolocation Data** - Geocoding and location-based services
- **Extension System** - Modular extension support for custom functionality

### Image Generation
- **Automatic1111 WebUI** integration for Stable Diffusion image generation

### Meta Features
- **PyInstaller** support for Windows executable distribution
- **Installer** creation with Inno Setup
- **MCP Tools** and servers for integration
- **GitHub Actions** deployment for web instance and releases
- **Extension System** for custom functionality

## Installation

### Quick Start
```bash
git clone https://github.com/TheCrazy8/Lunaa.git
cd Lunaa
pip install -r requirements.txt
python lunaa.py
```

### Windows Installation
1. Download the latest installer from [Releases](https://github.com/TheCrazy8/Lunaa/releases)
2. Run the installer
3. Launch Lunaa from the Start Menu or Desktop

## Available Commands

### Image & Media
- `/img <prompt> [|| negative]` - Generate image with Automatic1111
- `/vision <image_path> [question]` - Analyze image or answer questions about it

### Web & Data
- `/web <url or search>` - Search web or fetch URL content
- `/dataset load <name>` - Load HuggingFace dataset
- `/dataset query <name>` - Query loaded dataset

### Mathematics
- `/math <expression>` - Calculate mathematical expression
- `/plot <expression>` - Plot mathematical function and save as image

### Files
- `/file <path>` - View file contents
- `/dir [path]` - List directory contents

### Memory & Context
- `/memory add <fact>` - Add fact to memory
- `/memory search <query>` - Search memory for facts
- `/memory clear` - Clear all memory
- `/context` - View current context summary

### Location
- `/geo <address>` - Geocode an address to coordinates

### Extensions
- `/ext load <name>` - Load an extension
- `/ext unload <name>` - Unload an extension
- `/ext list` - List available extensions

### System
- `/sdstatus` - Check Automatic1111 WebUI status

## Building from Source

### Windows Executable
```bash
# Install PyInstaller
pip install pyinstaller

# Build
pyinstaller lunaa.spec
# Or use the build script
build_windows.bat
```

### Create Installer
1. Install [Inno Setup](https://jrsoftware.org/isinfo.php)
2. Open `installer_setup.iss` in Inno Setup
3. Compile to create installer

## Development

### Creating Extensions
Extensions are Python files placed in the `extensions/` directory:

```python
def initialize():
    """Return dict of command_name -> function"""
    return {
        'my_command': my_function
    }

def my_function(arg):
    """Your command implementation"""
    return f"Result: {arg}"

def cleanup():
    """Called when extension is unloaded"""
    pass
```

### MCP Tools
MCP (Model Context Protocol) servers are available for:
- Memory operations
- Vision analysis
- File operations

Configure in `mcp_config.json`

## Architecture

```
Lunaa/
├── lunaa.py                 # Main application
├── lunaa_modules/           # Core modules
│   ├── memory/             # Memory engine
│   ├── vision/             # Vision model
│   ├── audio/              # Audio processing
│   ├── data_sources/       # HuggingFace & APIs
│   ├── context/            # Context engine
│   ├── command_api/        # External API
│   ├── tools/              # Math, geo, files
│   └── extensions/         # Extension manager
├── extensions/             # User extensions
├── mcp_tools/             # MCP servers
└── .github/workflows/     # CI/CD pipelines
```

## Requirements

- Python 3.8+
- Ollama (for text model)
- Automatic1111 WebUI (optional, for image generation)
- See `requirements.txt` for Python packages

## Configuration

### Environment Variables
- `SD_API_URL` - Automatic1111 WebUI URL (default: http://127.0.0.1:7860)

### Memory Storage
Memory is stored in `lunaa_memory.json` in the working directory.

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Support

For issues and questions, please use the [GitHub Issues](https://github.com/TheCrazy8/Lunaa/issues) page.

