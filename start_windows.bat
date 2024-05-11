@echo off

call ollama create AwwWaifuAI -f models/modelfile.md

call python main.py

pause