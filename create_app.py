#!/bin/bash

# Create base project directory
mkdir relevancy_tagger
cd relevancy_tagger

# Create directory structure
mkdir -p models database search pages tests data config

# Create and populate files
cat > models/models.py << 'EOL'
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    id: str
    position: int
    title: str
    content: str
    relevancy_score: Optional[int] = None

@dataclass
class GoldenDataset:
    query: str
    results: List[SearchResult]
    evaluator: str
    date: str
EOL

cat > database/database.py << 'EOL'
from typing import List, Dict
from models.models import SearchResult

def save_feedback(query: str, result_id: str, position: int, is_relevant: bool) -> None:
    pass

def get_all_feedback() -> List[Dict]:
    pass

def get_feedback_stats() -> Dict:
    pass
EOL

cat > search/search.py << 'EOL'
from typing import List
from models.models import SearchResult

def search_results(query: str) -> List[SearchResult]:
    pass
EOL

cat > pages/search_page.py << 'EOL'
import streamlit as st
from database.database import save_feedback
from search.search import search_results

def render():
    st.title("Search Relevancy Tagging")
    
    query = st.text_input("Enter search query")
    
    if query:
        results = search_results(query)
        
        cols = st.columns(3)
        for idx, result in enumerate(results):
            with cols[idx % 3]:
                st.card(
                    title=result.title,
                    text=result.content,
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍", key=f"up_{result.id}"):
                        save_feedback(query, result.id, result.position, True)
                with col2:
                    if st.button("👎", key=f"down_{result.id}"):
                        save_feedback(query, result.id, result.position, False)
EOL

cat > pages/analysis_page.py << 'EOL'
import streamlit as st
from database.database import get_all_feedback, get_feedback_stats

def render():
    st.title("Feedback Analysis")
    
    stats = get_feedback_stats()
    feedback = get_all_feedback()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Feedback", stats.get("total", 0))
    with col2:
        st.metric("Relevant Results", stats.get("relevant", 0))
    with col3:
        st.metric("Non-Relevant Results", stats.get("non_relevant", 0))
    
    st.dataframe(feedback)
EOL

cat > main.py << 'EOL'
import streamlit as st
from pages import search_page, analysis_page

def main():
    st.set_page_config(layout="wide")
    
    page = st.sidebar.radio("Navigation", ["Search & Tag", "Analysis"])
    
    if page == "Search & Tag":
        search_page.render()
    else:
        analysis_page.render()

if __name__ == "__main__":
    main()
EOL

# Create config directory with sample config
mkdir -p config
cat > config/config.yaml << 'EOL'
database:
  path: "data/feedback.db"

search:
  endpoint: "http://localhost:8000/search"
  timeout: 5
EOL

# Create data directory with gitkeep
mkdir -p data
touch data/.gitkeep

# Create requirements.txt
cat > requirements.txt << 'EOL'
streamlit==1.32.0
pandas==2.2.0
pyyaml==6.0.1
SQLAlchemy==2.0.27
requests==2.31.0
python-dotenv==1.0.1
EOL

# Create empty __init__.py files
touch models/__init__.py
touch database/__init__.py
touch search/__init__.py
touch pages/__init__.py

# Create simple README
cat > README.md << 'EOL'
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
EOL

# Create basic .gitignore
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDEs
.idea/
.vscode/
*.swp
*.swo

# Project specific
data/*.db
.env
EOL

echo "Project structure created successfully!"
echo "Next steps:"
echo "1. Create virtual environment: python -m venv venv"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Install requirements: pip install -r requirements.txt"
echo "4. Run application: streamlit run main.py"
