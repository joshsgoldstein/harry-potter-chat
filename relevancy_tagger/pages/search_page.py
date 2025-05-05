import streamlit as st
from search.search import search_documents
from models.search_models import SearchResult
from search.feedback import save_feedback

def render():
    st.title("Search & Tag")
    
    query = st.text_input("Enter your search query:")
    
    if query:
        results = search_documents(query)
        
        if results:
            cols_per_row = 3
            num_results = len(results.objects)
            rows = num_results // cols_per_row + int(num_results % cols_per_row > 0)

            for row in range(rows):
                cols = st.columns(cols_per_row)
                for col_idx in range(cols_per_row):
                    result_idx = row * cols_per_row + col_idx
                    if result_idx >= num_results:
                        break
                    result = results.objects[result_idx]
                    position = result_idx + 1  # Position counter (1-based index)
                    print(position)
                    with cols[col_idx]:
                        st.markdown(f"### {result.properties['chapter_title']}")
                        st.write(result.properties["content"][:200] + "...")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("👍", key=f"up_{result.uuid}"):
                                save_feedback(query, result.uuid, position, True)  # Use position here
                        with col2:
                            if st.button("👎", key=f"down_{result.uuid}"):
                                save_feedback(query, result.uuid, position, False)  # Use position here
