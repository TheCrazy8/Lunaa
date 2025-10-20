import requests
import sys
import os
import random
import datetime
import tkinter as tk
import tkinter.ttk as ttk
import ollama
import sv-ttk
import numpy
from ollama import chat


ollama.pull('gemma3')

def chat():
    content=input("Messagetext here: ")
    stream = chat(
    model='gemma3',
    messages=[{'role': 'user', 'content': content}],
    stream=True,
    )

    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)

def vision():
    