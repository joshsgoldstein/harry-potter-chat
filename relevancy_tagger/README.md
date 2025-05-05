# Relevancy Tagger

A tool for creating golden datasets for search relevancy.

## Setup

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Project Structure

- `models/` - Data models
- `database/` - Database operations
- `search/` - Search integration
- `pages/` - Streamlit pages
- `config/` - Configuration files
- `data/` - Database and data files
- `tests/` - Test files
