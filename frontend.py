import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from src.contracts import run_query

st.title("KREPS RAG - Debug UI")

query = st.text_area("Enter your question:", height=100)
submit = st.button("Run Query")

if submit:
    if query.strip():
        st.write("**Query:**", query)
        st.write("---")

        try:
            st.write("Calling run_query()...")
            result = run_query(query.strip())

            st.write("**Raw Result:**")
            st.write(result)
            st.write("---")

            st.write("**Answer:**")
            st.write(result.get('answer', 'No answer'))

            st.write("**Confidence:**")
            st.write(result.get('confidence', 'Unknown'))

            st.write("**Sources:**")
            st.write(result.get('sources', []))

            st.write("**Chunks:**")
            st.write(result.get('chunks', []))

        except Exception as e:
            st.error("Exception occurred:")
            st.exception(e)
    else:
        st.warning("Please enter a question")
