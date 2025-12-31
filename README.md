# Easy eBook to Audiobook Converter with F5-TTS

![image](https://github.com/user-attachments/assets/d286b6e4-7a73-4e9d-88af-dc910652d743)


Turn your eBooks into audiobooks using the F5-TTS text-to-speech model. This application allows you to upload an eBook file and a reference voice sample to generate a personalized audiobook. The app supports various eBook formats and provides advanced settings to customize the output.

Copy and paste this single command line into command prompt to get the app running locally in Docker (Nvidia card accelerated)(11GB)(can take a long time to load, check CPU usage once download is finished, if CPU usage is high, it is loading correctly):

```bash
curl -L "https://huggingface.co/wildflowerdewdrop/f5tts_offline_ebook_to_audiobook_Docker_image/resolve/main/f5tts-app-preloaded_2025-12-30.tar?download=true" -o f5tts-app-preloaded_2025-12-30.tar && docker load < f5tts-app-preloaded_2025-12-30.tar && docker run --rm -it --gpus all -p 7860:7860 f5tts_custom:latest && del f5tts-app-preloaded_2025-12-30.tar

```

### Apple Silicon (M1/M2/M3/M4) Version

A prebuilt Apple Silicon version is available as a `.zip` download with Apple Silicon GPU (Metal) acceleration support:

[Download Apple Silicon F5-TTS.zip](https://huggingface.co/wildflowerdewdrop/f5tts_offline_ebook_to_audiobook_Docker_image/resolve/main/Apple%20Silicon%20f5tts%20backup%2012.30.2025.zip?download=true)

## Features

- **Voice Customization**: Upload a voice sample (<15 seconds) to mimic in the generated audiobook.
- Requires 5GB of Vram
- **Estimated Remaining Time**: See how much time is left to finish each ebook, calculated automatically
- **Multiple eBook Formats**: Supports `.epub`, `.mobi`, `.pdf`, `.txt`, and `.html` files.
- **Batch Processing**: Upload multiple eBooks for batch conversion.
- **Advanced Settings**:
  - Reference text input for more accurate voice cloning.
  - Adjust speech speed.
  - Customize cross-fade duration between audio chunks.
- **Metadata Handling**: Extracts and embeds eBook metadata and cover images into the audiobook files.
- **Output Formats**: Generates audiobooks with book covers embedded in `.mp3` format. (`.m4b` format caused play/pause issues in some audiobook players.)
- **User-Friendly Interface**: Built with Gradio for an interactive web UI.

## DEMO

https://github.com/user-attachments/assets/7dcc32f1-71be-4d99-945a-03d3eaf6b8ad


# How to Run the Offline F5 TTS Docker Application

These instructions will guide you through running the pre-packaged offline Text-to-Speech (TTS) Docker image using a single command. This image includes all necessary models, so it does not require an internet connection *after* the initial download step.

## Prerequisites

Before you begin, ensure you have the following installed and configured on your system:

1.  **Docker Desktop:** Install Docker for your operating system (Windows, macOS, or Linux). You can find it here: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)
2.  **NVIDIA GPU Drivers:** You need a compatible NVIDIA graphics card (GPU). Install the latest drivers from the NVIDIA website.
3.  **NVIDIA Container Toolkit:** This allows Docker to access your NVIDIA GPU. Installation instructions depend on your OS (primarily Linux, or WSL2 on Windows). Follow the official guide: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
4.  **Sufficient VRAM:** Ensure your NVIDIA GPU has at least **5 GB of Video RAM (VRAM)** available for the application to run smoothly.
5.  **`curl` Command-Line Tool:** This tool is used within the single command to download the image. It's pre-installed on most Linux and macOS systems, and usually included in modern Windows 10/11. You can verify it's available by opening your terminal/command prompt and typing `curl --version`.

*(Note: Running this application requires a compatible NVIDIA GPU meeting the VRAM requirement, properly configured for Docker with the NVIDIA Container Toolkit.)*

## Instructions

1.  **Open a Terminal or Command Prompt:**
    * **Windows:** Open Command Prompt, PowerShell, or the terminal in Windows Subsystem for Linux (WSL).
    * **macOS:** Open the Terminal application (found in Applications > Utilities).
    * **Linux:** Open your preferred terminal application.

2.  **Download, Load, and Run (Single Command):**
    * The single command below performs all the necessary steps sequentially: downloads the Docker image (~5.8 GB) from Hugging Face saving it temporarily, loads it into your Docker engine, starts the application container, and finally cleans up the downloaded file *after* you stop the container.
    * This method is more robust against download interruptions than piping the download directly.
    * Copy the entire command, paste it into your terminal.
    * **Command:**
        ```bash
        curl -L "https://huggingface.co/jdana/f5tts_offline_ebook_to_audiobook_Docker_image/resolve/main/f5tts-app-preloaded_2025-05-29.tar" -o f5tts-app-preloaded_2025-05-29.tar && docker load < f5tts-app-preloaded_2025-05-29.tar && docker tag 21fad7b5127e f5tts:latest && docker run --rm -it --gpus all -p 7860:7860 f5tts:latest && del f5tts-app-preloaded_2025-05-29.tar
        
        ```
        *(Note: On Linux/macOS, change the final `del` command to `rm`)*
    * **How it works:**
        * `curl -L "URL" -o filename.tar`: Downloads the file completely and saves it locally. `-L` follows redirects, `-o` specifies the output filename.
        * `&&`: Ensures the next command runs *only if* the previous one was successful.
        * `docker load < filename.tar`: Loads the image into Docker from the downloaded file.
        * `&&`: Ensures the `docker run` command starts *only if* the load was successful.
        * `docker run ...`: Runs the container using the `f5tts:latest` image. The `--rm` flag ensures the container is removed when stopped. `-it` runs it interactively. `--gpus all` provides GPU access. `-p 7860:7860` maps the port.
        * `&&`: Ensures the cleanup command runs *only if* `docker run` exits successfully (i.e., after you stop the container, typically with `Ctrl+C`).
        * `del filename.tar` (or `rm filename.tar` on Linux/macOS): Deletes the downloaded `.tar` file to save space.
    * **Be Patient:** This command still downloads a large file (11 GB), which will take time based on your internet speed. You will see download progress from `curl`. The subsequent `docker load` process also takes time. Wait for the command to complete the download and load steps before the application logs start appearing.

## Accessing the Application

* Once the container is running (you should eventually see log messages in the terminal, including something like `* Running on local URL: http://0.0.0.0:7860`), open your web browser.
* Navigate to the following address:
    **`http://localhost:7860`**
* You should now see the web interface for the TTS application.

## Stopping the Application

* To stop the application container, go back to the terminal window where it is running.
* Press `Ctrl + C` (hold the Control key and press C).
* The container will stop. Because the run command included the `--rm` flag, it will also be automatically removed. The final part of the single command (`del` or `rm`) should then execute to clean up the downloaded `.tar` file.

# Running f5tts Locally with Docker

These instructions guide you on how to download the f5tts Docker image once, load it into your local Docker repository, and then run it anytime without re-downloading.

## Prerequisites

*   **Docker Desktop** (or Docker Engine) installed and running on your system.
*   **Command Prompt** (or PowerShell).
*   (Optional but likely required for this image) **NVIDIA GPU drivers** and the **NVIDIA Container Toolkit** if you plan to use GPU acceleration (`--gpus all` flag).

## Phase 1: One-Time Setup

Do these steps only once, or when you want to update to a new version of the image downloaded as a `.tar` file.

1.  **Open Command Prompt** (it's often a good idea to run it as Administrator for Docker operations, though not always strictly necessary).

2.  **Navigate to a directory** where you want to temporarily store the downloaded image archive (e.g., `C:\DockerImages`):
    ```cmd
    cd C:\path\to\your\desired\folder
    ```

3.  **Download the Docker image archive:**
    ```cmd
    curl -L "https://huggingface.co/jdana/f5tts_offline_ebook_to_audiobook_Docker_image/resolve/main/f5tts-app-preloaded_2025-05-29.tar" -o f5tts-app-preloaded_2025-05-29.tar
    ```
    *   This will download the `f5tts-app-preloaded_2025-05-29.tar` file. Wait for the download to complete as it can be a large file.

4.  **Load the image from the archive into Docker:**
    ```cmd
    docker load < f5tts-app-preloaded_2025-05-29.tar
    ```
    *   This command extracts and loads the image layers into Docker's local storage. It might take some time.
    *   Upon completion, it usually outputs the ID of the loaded image, for example: `Loaded image ID: sha256:21fad7b5127e...`.

5.  **Tag the loaded image with a friendly name:**
    The image ID `21fad7b5127e` is the short ID for the image within this specific tarball.
    ```cmd
    docker tag 21fad7b5127e f5tts:latest
    ```
    *   This assigns the more memorable tag `f5tts:latest` to the image.
    *   You can verify the image is tagged correctly by running `docker images` and looking for `f5tts` in the `REPOSITORY` column.

6.  **(Optional) Delete the downloaded `.tar` file** to save disk space, as the image is now stored by Docker:
    ```cmd
    del f5tts-app-preloaded_2025-05-29.tar
    ```

## Phase 2: Running the Application Locally

Do these steps every time you want to use the f5tts application.

1.  **Open Command Prompt.**

2.  **Run the Docker container** using the tag you created:
    ```cmd
    docker run --rm -it --gpus all -p 7860:7860 f5tts:latest
    ```
    *   `--rm`: Automatically removes the container when it exits.
    *   `-it`: Runs in interactive mode and allocates a pseudo-TTY.
    *   `--gpus all`: (Requires NVIDIA setup) Exposes all available NVIDIA GPUs to the container. If you don't have an NVIDIA GPU or the NVIDIA Container Toolkit, you might need to remove this flag, but the application's performance or functionality could be affected.
    *   `-p 7860:7860`: Maps port 7860 on your host machine to port 7860 inside the container. This allows you to access the application via `http://localhost:7860` in your web browser.
    *   `f5tts:latest`: Specifies the Docker image to run.

Once the container starts, you should be able to access the f5tts web interface by navigating to `http://localhost:7860` in your browser. To stop the container, press `Ctrl+C` in the Command Prompt window where it's running.

## License:

-   GPL-3.0

## Acknowledgments

-   This project uses code adapted from [fakerybakery](https://github.com/fakerybakery)'s Hugging Face space [E2-F5-TTS](https://huggingface.co/spaces/mrfakename/E2-F5-TTS) and [DrewThomasson](https://github.com/DrewThomasson)'s Hugging Face space [ebook2audiobook](https://huggingface.co/spaces/drewThomasson/ebook2audiobook). Thanks for your amazing work!
