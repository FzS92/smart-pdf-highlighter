# Smart PDF Highlighter

Welcome to Smart PDF Highlighter! This tool automatically identifies and highlights important content within your PDF files. It utilizes many AI techniques such as deep learning and other advanced algorithms to analyze the text and intelligently select key sentences for highlighting.

## Overview

![ScreenShot](./photos/app.png)

The Smart PDF Highlighter functions with the following workflow:

1. **User Interface**: Users interact with the Streamlit-based graphical user interface (GUI) to upload their PDF files.
2. **PDF Processing**: Upon file upload, the tool processes the PDF content to identify important sentences.
3. **Highlighting**: Important sentences are highlighted within the PDF, emphasizing key content.
4. **Download**: Users can download the highlighted PDF for further reference.

## Installation

To use the Smart PDF Highlighter, follow these simple steps:

1. **Clone the Repository:** Clone the repository to your local machine.
    ```python
    git clone https://github.com/your-username/smart-pdf-highlighter.git
    cd smart-pdf-highlighter
    ```

2. **Create Virtual Environment:** Set up a Python 3.8 virtual environment and activate it.
    ```python
    conda create -n smart-pdf-env python=3.8
    conda activate smart-pdf-env
    ```

3. **Install Requirements:** Install the required dependencies.
    ```python
    pip install -r requirements.txt
    ```

## Usage

Follow these steps to run the Smart PDF Highlighter:

1. **Run the Application:** Execute the `app.py` script to start the Streamlit application.
    ```python
    streamlit run app.py
    ```

2. **Upload PDF:** Use the provided interface to upload your PDF file.

3. **Highlighting:** Once the file is uploaded, the tool will automatically process it and generate a highlighted version.

4. **Download:** Download the highlighted PDF using the provided download button.




