# Obesity App

## Project Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Create a virtual environment**
   ```bash
   py -m venv .venv
   ```

2. **Activate the virtual environment**
   ```bash
   .venv/Scripts/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Activate the virtual environment** (if not already activated)
   ```bash
   .venv/Scripts/activate
   ```

2. **Run the Streamlit application**
   ```bash
   streamlit run app.py
   ```

The application will open in your default web browser at `http://localhost:8501`.

## Deactivating the Virtual Environment

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

## Troubleshooting

- If you encounter permission issues on Windows, try running the terminal as Administrator
- If `py` command is not recognized, try using `python` instead
- Make sure you're in the project directory when running these commands
