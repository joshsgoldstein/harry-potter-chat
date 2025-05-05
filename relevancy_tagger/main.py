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
