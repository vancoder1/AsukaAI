# AsukaAI

AsukaAI is a fully offline, local AI solution designed to ensure privacy while providing robust functionalities. It integrates speech-to-text, text generation, and text-to-speech capabilities using the following technologies:
- **faster_whisper** for speech-to-text
- **ollama** with Llama3 under the hood for text generation
- **coqui_tts** for text-to-speech

## 🌟 Features

- **🔒 Privacy**: All processing is done locally on your machine, ensuring your data never leaves your device.
- **🛠️ Customizability**: Create and integrate your own models according to your needs.
- **🔗 Modular Design**: Each component (STT, text generation, TTS) can be independently configured and customized.
- **👥 User-friendly**: Easy to set up and use with straightforward installation and configuration.

## 🖥️ System Requirements

### Minimum Requirements:
- **VRAM**: 0GB (GPU not required)
- **RAM**: 16GB
- **Free Disk Space**: 10GB
- **OS**: Windows

### Recommended Requirements:
- **VRAM**: 12GB
- **RAM**: 16GB
- **Free Disk Space**: 10GB
- **OS**: Windows

#### Notes:
- **GPU**: Nvidia GPU (CPU if Nvidia GPU is not available).
- **AMD GPUs are not supported yet**.

## 🚀 Installation

### Prerequisites

1. **Python 3.11**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **CUDA Toolkit** (if using GPU): Ensure you have the CUDA toolkit installed to leverage GPU acceleration. Download it from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit).

3. **Ollama**: Ensure Ollama is installed on your system. Download it here [Ollama](https://ollama.com/).

4. **Miniconda**: Ensure Miniconda is installed. Download it here [Miniconda](https://docs.anaconda.com/free/miniconda/index.html)

### Steps

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/vancoder1/AsukaAI.git
    cd AsukaAI
    ```

2. **Install everything using install_windows.bat**:
    ```sh
    .\install_windows.bat
    ```

3. **Wait for the installation to complete**.

## 📖 Usage

1. **Start the Application**:
    ```sh
    .\start_windows.bat
    ```

2. **Speech-to-Text**:
    - Push the default F8 button (which can be rebound within the program) to start recording.
    - Wait for faster_whisper to process and extract text from your recording.

3. **Text Generation**:
    - Ollama3 provides human-like interaction. If you prefer a different model, you can change it in `models/modelfile.md`.
    - Check the [Ollama library](https://ollama.com/library) for available models.
    - For customizing model behavior, refer to the Ollama documentation on their [GitHub page](https://github.com/ollama/ollama).
    - **Important**: After modifying `models/modelfile.md`, run the following command to update the model:
      ```sh
      .\update_model.bat
      ```

4. **Text-to-Speech**:
    - Convert generated text to speech using the `XTTS_v2` model in `coqui_tts`.
    - To change the voice, place your desired WAV audio file (approximately 15 seconds, clear voice, no background noise) into the `xtts_voices/` directory and name the file `input.wav`.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgements

- [faster_whisper](https://github.com/guillaumekln/faster-whisper)
- [ollama](https://github.com/ollama/ollama)
- [coqui_tts](https://github.com/coqui-ai/TTS)

## 📬 Contact

For any questions or feedback, please open an issue on this repository or reach out to `ivanzaporozhets25@gmail.com`.
