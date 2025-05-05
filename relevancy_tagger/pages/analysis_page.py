import streamlit as st
from search.feedback import get_all_feedback
import pandas as pd

def render():
    st.title("Feedback Analysis")
    
    # Check if feedback is already in session state
    if 'feedback' not in st.session_state:
        st.session_state.feedback = get_all_feedback()
    
    feedback = st.session_state.feedback
    print(feedback)

    if not feedback:  # Check if feedback is empty
        st.write("No feedback available.")
        return  # Exit the function early

    # Prepare data for the DataFrame
    data = []
    for index, item in feedback.items():
        properties = item.properties  # Access the properties of the object
        data.append({
            "UUID": str(item.uuid),  # Convert UUID to string for display
            "Feedback Type": properties.get('feedback_type'),
            "Position": properties.get('position'),
            "Query": properties.get('query'),
            "Document ID": properties.get('document_id'),
            "Select": st.checkbox(f"Select {properties.get('query')}", key=index)  # Checkbox for selection
        })

    # Create a DataFrame from the data
    df = pd.DataFrame(data)

    # Display the DataFrame as an interactive table
    st.dataframe(df)

    # Optionally, handle selected feedback
    selected_feedback = df[df['Select']].copy()  # Get selected rows
    if not selected_feedback.empty:
        st.write("Selected Feedback for Golden Dataset:")
        st.table(selected_feedback)  # Display selected feedback as a table
