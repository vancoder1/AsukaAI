# AsukaAI

AsukaAI is your personal offline AI companion, ensuring privacy while offering powerful features. This local AI waifu seamlessly integrates speech-to-text, text generation, and text-to-speech functionalities:
- **RealtimeSTT** with faster_whisper under the hood for speech-to-text
- **Ollama** with Llama3 under the hood for text generation
- **RealtimeTTS** with CoquiTTS under the hood for text-to-speech

## 🌟 Features

- **Privacy**: All processing is done locally on your machine, ensuring your data never leaves your device.
- **Customizability**: Create and integrate your own models according to your needs.
- **Modular Design**: Each component (STT, text generation, TTS) can be independently configured and customized.
- **User-friendly**: Easy to set up and use with straightforward installation and configuration.
- **Real-time processing**: Speech-to-text and text-to-speech are performed in real-time, enabling almost seamless interaction with AI.

## 🖥️ System Requirements

### Minimum Requirements:
- **VRAM**: 0GB (GPU not required)
- **RAM**: 16GB
- **Free Disk Space**: 10GB

### Recommended Requirements:
- **VRAM**: 12GB
- **RAM**: 16GB
- **Free Disk Space**: 10GB

#### Notes:
- **GPU**: Nvidia GPU recommended; CPU can be used if Nvidia GPU is not available.
- **AMD GPUs** Not supported yet.
- **OS**: Windows is preferable; Linux and macOS have not been tested yet.

## 🚀 Installation

### Prerequisites

1. **Python 3.11**: Make sure Python is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **CUDA Toolkit** (if using GPU): Ensure you have the CUDA toolkit installed to leverage GPU acceleration. Download it from [NVIDIA's website](https://developer.nvidia.com/cuda-toolkit).

3. **Ollama**: Ensure Ollama is installed on your system. Download it here [Ollama](https://ollama.com/).

4. **Miniconda**: Ensure Miniconda is installed. Don't forget to check `Add to PATH` during installation. Download it here [Miniconda](https://docs.anaconda.com/free/miniconda/index.html).

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
    - Just start speaking to start transcription.
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
    - Convert generated text to speech in real time while text is being generated.
    - To change the voice, place your desired WAV audio file (approximately 10 seconds, clear voice, no background noise, 22050hz, mono) into the `data/reference_voices` directory and name the file `reference.wav`. Make sure you removed previous `reference.wav` and `reference.json` files.

5. **Configuration**:
    - You can customize various settings in the `config.json` file:
    ```json
    {
        "version": "1.0.0",
        "description": "Configuration file for AI app",

        "stt": {
            "model": "distil-small.en"
        },

        "tts": {
            "reference_file": "data/reference_voices/reference.wav"
        }
    }
    ```
    - **stt.model**: Defines the model used for speech-to-text processing. You can change this to other available models such as `"base.en"` or `"distil-large-v3"` etc.
    - **tts.reference_file**: Sets the path to the reference audio file for text-to-speech processing. Ensure the specified file exists and is correctly formatted.

## 🤝 Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the [Apache License 2.0](LICENSE).

## 🙏 Acknowledgements

- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- [Ollama](https://github.com/ollama/ollama)
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)

## 📬 Contact

For any questions or feedback, please open an issue on this repository or reach out to `ivanzaporozhets25@gmail.com`.
