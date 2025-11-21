# Lunaa AI Commands Reference

## Image Generation
### /img <prompt> [|| negative]
Generate an image using Automatic1111 Stable Diffusion.

**Examples:**
- `/img a beautiful sunset over mountains`
- `/img a cat wearing a hat || blurry, low quality`

## Vision & Image Analysis
### /vision <image_path> [question]
Analyze an image using the BLIP vision model.

**Examples:**
- `/vision photo.jpg` - Get general description
- `/vision photo.jpg What color is the sky?` - Ask specific question

## Web Search & Scraping
### /web <url or search terms>
Search the web using DuckDuckGo or fetch a specific URL.

**Examples:**
- `/web latest AI news`
- `/web https://example.com`

## Mathematics
### /math <expression>
Calculate a mathematical expression.

**Examples:**
- `/math 2 + 2 * 3`
- `/math sqrt(16) + log(10)`
- `/math sin(pi/2)`

### /plot <expression>
Plot a mathematical function and save as image.

**Examples:**
- `/plot sin(x)`
- `/plot x**2 + 2*x + 1`
- `/plot exp(-x**2)`

## File Operations
### /file <filepath>
View the contents of a file.

**Examples:**
- `/file README.md`
- `/file config.json`

### /dir [path]
List directory contents.

**Examples:**
- `/dir` - List current directory
- `/dir /home/user/documents`

## Memory Management
### /memory add <fact>
Add a fact to persistent memory.

**Example:**
- `/memory add My favorite color is blue`

### /memory search <query>
Search memory for facts matching query.

**Example:**
- `/memory search favorite`

### /memory clear
Clear all memory.

## Datasets (HuggingFace)
### /dataset load <name>
Load a dataset from HuggingFace.

**Example:**
- `/dataset load imdb`

### /dataset query <name>
Query a loaded dataset.

**Example:**
- `/dataset query imdb`

## Geolocation
### /geo <address>
Geocode an address to coordinates.

**Examples:**
- `/geo New York City`
- `/geo 1600 Pennsylvania Avenue`

## Extensions
### /ext load <name>
Load an extension.

**Example:**
- `/ext load example_extension`

### /ext unload <name>
Unload an extension.

**Example:**
- `/ext unload example_extension`

### /ext list
List all available extensions.

## Context & System
### /context
View current conversation context summary.

### /sdstatus
Check if Automatic1111 WebUI is online.

## Tips
- Commands are case-sensitive and start with `/`
- Use quotes for paths with spaces
- Some features require additional dependencies (see requirements.txt)
- Extensions can add custom commands

## Getting Help
- Type `/help` to see this reference
- Visit the GitHub repository for detailed documentation
- Check the README for installation and setup instructions
