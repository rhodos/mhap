# mhap
MHAP stands for Mental Health App Profiler. It includes code to extract and analyze information about mental health-related apps. 

To setup your environment to run the code, you must do the following:
* Change BASE_DIR in .env file to point to your project directory.
* Add the mhap/code directory to your PYTHONPATH, e.g. by adding the following to your .bashrc or .zshrc file (be sure to replace the path with your own):
  ```
  export PYTHONPATH="/path/to/mhap/code:$PYTHONPATH"
  ```
* Get an API key from OpenAI and add it to your environment as OPENAI_API_KEY
* (If you want to run app_classification_huggingface.py) Get a token from HuggingFace and add it to your environment as HF_TOKEN. Also, depending on which model you try to run, you may need to request permission via the HuggingFace website to use that model. Note that this script is exploratory and was not used in the final dataset construction.
* Setup and activate a virtualenv for your project, e.g. using pyenv:
  ```
    pyenv virtualenv python.3.12.11 mhap_venv
    pyenv activate mhap_venv
  ```
* Install the necessary requirements:
  ```
  pip install requirements.txt
  ```
