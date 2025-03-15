import streamlit as st
import os
import json
from src.knowledge_base import sync_github_knowledge_base

def github_connector_ui():
    """Renders the GitHub connector UI component in Streamlit."""
    
    st.markdown("## GitHub Knowledge Base")
    
    # Load saved configuration if any
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "config",
        "github_config.json"
    )
    
    # Initialize configuration
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            config = {"repo_url": "", "branch": "main"}
    else:
        config = {"repo_url": "", "branch": "main"}
    
    # Repository settings
    repo_url = st.text_input(
        "GitHub Repository URL", 
        value=config.get("repo_url", ""),
        placeholder="https://github.com/username/repository"
    )
    
    col1, col2 = st.columns([3, 2])
    with col1:
        branch = st.text_input(
            "Branch", 
            value=config.get("branch", "main")
        )
    
    with col2:
        use_azure = st.checkbox("Use Azure for Embeddings", value=True)
    
    # Save button for configuration
    if st.button("Save Configuration"):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        config = {
            "repo_url": repo_url,
            "branch": branch
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        st.success("Configuration saved!")
    
    # Sync button
    if st.button("Sync GitHub Knowledge Base"):
        if not repo_url:
            st.error("Please enter a GitHub repository URL")
        else:
            with st.spinner("Syncing repository and updating knowledge base..."):
                try:
                    stats, processed_files = sync_github_knowledge_base(
                        repo_url=repo_url,
                        branch=branch,
                        use_azure=use_azure
                    )
                    
                    # Store processed files in session state
                    st.session_state.github_processed_files = processed_files
                    st.session_state.github_sync_expanded = True
                    
                    # Show results
                    st.success("Sync completed successfully!")
                    st.write(f"New files: {stats['new_files']}")
                    st.write(f"Changed files: {stats['changed_files']}")
                    st.write(f"Documents processed: {stats['processed_documents']}")
                    st.write(f"Last sync: {stats['last_sync']}")
                    
                    # Force refresh of the app to reflect changes
                    if stats['new_files'] > 0 or stats['changed_files'] > 0:
                        st.session_state.clear_langchain = True
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error syncing repository: {str(e)}")
    
    # Display processed files from the last sync
    # Instead of using an expander, use a markdown toggle section
    if "github_processed_files" in st.session_state and st.session_state.github_processed_files:
        st.markdown("### Files added to vector storage")
        
        # Add a show/hide button using checkbox
        show_files = st.checkbox("Show file details", 
                                value=st.session_state.get("github_sync_expanded", False))
        
        if show_files:
            # Create a container for the files
            file_container = st.container()
            with file_container:
                for i, file_info in enumerate(st.session_state.github_processed_files):
                    filename = os.path.basename(file_info["path"])
                    relative_path = file_info["relative_path"]
                    status = file_info["status"]
                    
                    # Format the display with color based on status
                    if status == "new":
                        st.markdown(f"🆕 **{filename}** - {relative_path}")
                    elif status == "updated":
                        st.markdown(f"🔄 **{filename}** - {relative_path}")
                    else:
                        st.markdown(f"📄 **{filename}** - {relative_path}")

    # Display last sync info if available
    tracking_files = []
    github_repos_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "github_repos"
    )
    
    if os.path.exists(github_repos_dir):
        for file in os.listdir(github_repos_dir):
            if file.endswith("_tracking.json"):
                tracking_files.append(os.path.join(github_repos_dir, file))
    
    if tracking_files:
        st.markdown("### Sync History")
        for tracking_file in tracking_files:
            try:
                with open(tracking_file, 'r') as f:
                    data = json.load(f)
                    st.write(f"Repository: {data['repo_url']}")
                    st.write(f"Last sync: {data['last_sync']}")
                    st.write(f"Files tracked: {len(data['files'])}")
                    st.write("---")
            except:
                continue