# Lunaa AI Usage Examples

This document provides practical examples of using Lunaa AI's features.

## Getting Started

Launch Lunaa:
```bash
python lunaa.py
```

## Basic Chat
Simply type your message and press Enter:
```
You: What is the capital of France?
```

## Image Generation
Generate images with Automatic1111 (requires A1111 WebUI running):
```
/img a serene lake at sunset with mountains in the background

/img a cyberpunk city || blurry, low quality
```

## Web Search
Search the web or fetch specific URLs:
```
/web latest Python 3.12 features

/web https://github.com/TheCrazy8/Lunaa
```

## Vision & Image Analysis
Analyze images using the BLIP vision model:
```
/vision photo.jpg

/vision landscape.png What time of day is shown in this image?
```

## Mathematics
### Simple Calculations
```
/math 15 * 23 + 100

/math sqrt(144)

/math sin(pi/4) * cos(pi/4)
```

### Plotting Functions
```
/plot sin(x) * exp(-x/10)

/plot x**3 - 5*x**2 + 3*x + 1
```

## File Operations
### View Files
```
/file README.md

/file config.json

/file /path/to/document.txt
```

### List Directories
```
/dir

/dir /home/user/projects

/dir C:\Users\Username\Documents
```

## Memory Management
### Add Facts
```
/memory add My birthday is January 15th

/memory add I prefer Python for data science projects

/memory add The project deadline is next Friday
```

### Search Memory
```
/memory search birthday

/memory search Python

/memory search deadline
```

### Clear Memory
```
/memory clear
```

## HuggingFace Datasets
### Load Datasets
```
/dataset load imdb

/dataset load squad

/dataset load wikitext
```

### Query Datasets
```
/dataset query imdb
```

## Geolocation
Get coordinates for addresses:
```
/geo San Francisco, CA

/geo 1600 Pennsylvania Avenue, Washington DC

/geo Times Square, New York
```

## Extensions
### List Available Extensions
```
/ext list
```

### Load an Extension
```
/ext load example_extension
```

### Unload an Extension
```
/ext unload example_extension
```

## Context Management
View current conversation context:
```
/context
```

## System Commands
### Check Automatic1111 Status
```
/sdstatus
```

### Get Help
```
/help
```

## Advanced Usage

### Combining Features
You can combine multiple features in a workflow:

1. Search for information:
   ```
   /web Python async programming best practices
   ```

2. Save important facts:
   ```
   /memory add Async programming uses async/await keywords
   ```

3. Later, recall the information:
   ```
   /memory search async
   ```

### Working with Files
1. List project directory:
   ```
   /dir /home/user/my_project
   ```

2. View a specific file:
   ```
   /file /home/user/my_project/main.py
   ```

3. Ask about the code:
   ```
   You: Can you explain the main.py file?
   ```

### Mathematical Analysis
1. Calculate values:
   ```
   /math log(1000)
   ```

2. Plot the function:
   ```
   /plot log(x)
   ```

3. Analyze the visualization (if saved to plot.png):
   ```
   /vision plot.png Describe this mathematical function
   ```

## Tips & Best Practices

1. **Use descriptive image prompts**: More detailed prompts generate better images
2. **Memory management**: Regularly use `/memory search` to recall important information
3. **File paths**: Use absolute paths for reliability across different directories
4. **Web searches**: Be specific in your search queries for better results
5. **Extensions**: Create custom extensions for repeated tasks
6. **Context**: Use `/context` to understand what information the AI is tracking

## Troubleshooting

### Image Generation Not Working
- Ensure Automatic1111 WebUI is running
- Check `/sdstatus` to verify connection
- Verify SD_API_URL environment variable if using custom port

### Module Not Available Errors
- Install missing dependencies: `pip install -r requirements.txt`
- Some features require optional packages (transformers, datasets, etc.)

### Memory Not Persisting
- Check file permissions for `lunaa_memory.json`
- Ensure you have write permissions in the working directory

## Creating Custom Extensions

Create a file in `extensions/my_extension.py`:

```python
def initialize():
    return {
        'greet': greet_user,
        'count': count_words
    }

def greet_user(name):
    return f"Hello, {name}! Welcome to my extension."

def count_words(text):
    return f"Word count: {len(text.split())}"

def cleanup():
    print("Extension unloaded")
```

Load it with:
```
/ext load my_extension
```

## More Information

- See `README.md` for installation and setup
- See `COMMANDS.md` for complete command reference
- Visit GitHub for issues and contributions
