import streamlit as st
import os
import pandas as pd
from src.embeddings import (
    get_all_documents_metadata, 
    delete_documents_by_ids,
    delete_documents_by_source
)

def vector_store_manager_ui():
    """UI component for managing the vector store contents."""
    st.markdown("## Vector Store Management")
    
    # Get vector store path
    vector_store_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "vector_store", 
        "index"
    )
    
    # Check if vector store exists
    if not os.path.exists(vector_store_path):
        st.warning("Vector store not found. Please add some documents first.")
        return
    
    # Get documents metadata
    with st.spinner("Loading vector store contents..."):
        documents = get_all_documents_metadata(vector_store_path)
    
    if not documents:
        st.info("No documents found in vector store.")
        return
    
    # Display document count
    st.write(f"Found {len(documents)} document chunks in vector store.")
    
    # Transform to more readable format
    display_data = []
    sources = set()
    
    for doc in documents:
        source = doc.get('source', 'unknown')
        filename = doc.get('filename', os.path.basename(source) if source != 'unknown' else 'unknown')
        sources.add(source)
        
        display_data.append({
            'ID': doc.get('id', 'unknown'),
            'Source': source,
            'Filename': filename,
            'Type': doc.get('filetype', 'unknown'),
            'Preview': doc.get('content_preview', '')[:50]
        })
    
    # Group by source for summary
    st.markdown("### Document Sources")
    source_counts = {}
    for doc in documents:
        source = doc.get('source', 'unknown')
        if source in source_counts:
            source_counts[source] += 1
        else:
            source_counts[source] = 1
    
    # Create a DataFrame for better display
    sources_df = pd.DataFrame({
        'Source': list(source_counts.keys()),
        'Chunks': list(source_counts.values())
    }).sort_values(by='Chunks', ascending=False)
    
    st.dataframe(sources_df, use_container_width=True)
    
    # Document deletion options
    st.markdown("### Delete Documents")
    delete_option = st.radio(
        "Delete by:", 
        ["Source File", "Individual Documents"],
        horizontal=True
    )
    
    if delete_option == "Source File":
        # Delete by source
        source_to_delete = st.selectbox(
            "Select source to delete", 
            sorted(list(sources))
        )
        
        if st.button("Delete Source", type="primary"):
            with st.spinner(f"Deleting documents from {source_to_delete}..."):
                success, count = delete_documents_by_source(
                    vector_store_path, 
                    source_to_delete
                )
                
                if success:
                    st.success(f"Successfully deleted {count} document chunks from {source_to_delete}")
                    # Clear the cache to force reloading the vector store
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Failed to delete documents from {source_to_delete}")
    else:
        # Show detailed documents view
        st.markdown("### Document Details")
        
        # Create DataFrame for display
        df = pd.DataFrame(display_data)
        
        # Display with selection
        selection = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.TextColumn("ID", width="small"),
                "Preview": st.column_config.TextColumn("Preview", width="large"),
            },
            disabled=["ID", "Source", "Filename", "Type", "Preview"],
            num_rows="dynamic"
        )
        
        # Get selected rows (if any)
        if len(selection) > 0:
            selected = st.multiselect(
                "Select documents to delete",
                options=df['ID'].tolist(),
                format_func=lambda x: f"{x[:8]}... - {df[df['ID'] == x]['Filename'].iloc[0]}"
            )
            
            if selected and st.button("Delete Selected", type="primary"):
                with st.spinner(f"Deleting {len(selected)} document chunks..."):
                    success = delete_documents_by_ids(vector_store_path, selected)
                    
                    if success:
                        st.success(f"Successfully deleted {len(selected)} document chunks")
                        # Clear the cache to force reloading the vector store
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error("Failed to delete selected documents")
