import os
import streamlit as st
from typing import List, Optional
import json
from src.github_sync import GitHubSyncManager
from src.document_loader import load_document
from src.embeddings import chunk_documents, add_documents_to_vector_store

def sync_github_knowledge_base(
    repo_url: str, 
    branch: str = "main",
    use_azure: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> tuple:
    """
    Synchronize a GitHub repository's markdown content with the knowledge base.
    
    Args:
        repo_url: The GitHub repository URL
        branch: The branch to synchronize (default: main)
        use_azure: Whether to use Azure for embeddings
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between text chunks
        
    Returns:
        Tuple containing stats dict and list of processed file details
    """
    # Create a temporary directory for syncing
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sync_dir = os.path.join(project_root, "github_repos")
    os.makedirs(sync_dir, exist_ok=True)
    
    # Initialize the GitHub sync manager
    sync_manager = GitHubSyncManager(repo_url=repo_url, local_dir=None, branch=branch)
    
    # Sync the repository and get new/changed files
    new_count, changed_count, updated_files = sync_manager.sync_repository()
    
    # If there are any files to process, add them to the vector store
    processed_count = 0
    processed_files = []
    
    if updated_files:
        documents = []
        
        # Load all updated markdown files
        for file_path in updated_files:
            try:
                # Determine if file is new or changed
                relative_path = os.path.relpath(
                    file_path, 
                    os.path.join(sync_dir, repo_url.split("/")[-1].replace(".git", ""))
                )
                
                # Check if this is a new or updated file
                local_dir = sync_manager.local_dir
                base_dir = os.path.dirname(local_dir)
                repo_name = os.path.basename(local_dir)
                tracking_file = os.path.join(base_dir, f"{repo_name}_tracking.json")
                
                status = "unknown"
                if os.path.exists(tracking_file):
                    try:
                        with open(tracking_file, 'r') as f:
                            tracking_data = json.load(f)
                            if relative_path not in tracking_data.get("files", {}):
                                status = "new"
                            else:
                                status = "updated"
                    except:
                        pass
                
                # Load the document
                file_docs = load_document(file_path)
                documents.extend(file_docs)
                processed_count += len(file_docs)
                
                # Add to processed files list
                processed_files.append({
                    "path": file_path,
                    "relative_path": relative_path,
                    "status": status,
                    "document_count": len(file_docs)
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
        
        if documents:
            # Process and add to vector store
            chunks = chunk_documents(
                documents, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            
            # Get vector store path and add documents
            vs_path = os.path.join(project_root, "vector_store", "index")
            add_documents_to_vector_store(chunks, vs_path, use_azure=use_azure)
    
    # Return stats about the operation and file details
    stats = {
        "new_files": new_count,
        "changed_files": changed_count,
        "processed_documents": processed_count,
        "last_sync": sync_manager.tracking_data.get("last_sync")
    }
    
    return stats, processed_files