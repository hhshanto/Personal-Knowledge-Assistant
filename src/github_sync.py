import os
import json
import shutil
import tempfile
import hashlib
from git import Repo, GitCommandError
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class GitHubSyncManager:
    """Manages syncing markdown files from a GitHub repository."""
    
    def __init__(self, repo_url: str, local_dir: str = None, branch: str = "main"):
        """
        Initialize the GitHub sync manager.
        
        Args:
            repo_url: URL of the GitHub repository
            local_dir: Directory to store the repository locally (if None, uses temp dir)
            branch: Branch to clone/pull from
        """
        self.repo_url = repo_url
        self.branch = branch
        self.repo_name = repo_url.split("/")[-1].replace(".git", "")
        
        # Use provided directory or create one in the project
        if local_dir:
            self.local_dir = local_dir
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.local_dir = os.path.join(project_root, "github_repos", self.repo_name)
            
        # Path to tracking file
        self.tracking_file = os.path.join(
            os.path.dirname(self.local_dir), 
            f"{self.repo_name}_tracking.json"
        )
        
        # Initialize tracking data
        self.tracking_data = self._load_tracking_data()
        
    def get_file_status(self, file_path):
        """Determine if a file is new or modified."""
        relative_path = os.path.relpath(file_path, self.local_dir)
        
        if relative_path not in self.tracking_data["files"]:
            return "new"
        else:
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != self.tracking_data["files"][relative_path]:
                return "modified"
            return "unchanged"
    

    def _load_tracking_data(self) -> Dict:
        """Load the tracking data from file or create a new one."""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading tracking file. Creating new tracking data.")
        
        # Initialize new tracking data
        return {
            "repo_url": self.repo_url,
            "last_commit": None,
            "last_sync": None,
            "files": {}  # Will store file paths and their MD5 hashes
        }
    
    def _save_tracking_data(self) -> None:
        """Save tracking data to file."""
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate MD5 hash of a file."""
        with open(file_path, 'rb') as file:
            return hashlib.md5(file.read()).hexdigest()
    
    def sync_repository(self) -> Tuple[int, int, List[str]]:
        """
        Sync the repository and identify new/changed files.
        
        Returns:
            Tuple containing (
                number of new files, 
                number of changed files,
                list of paths to new/changed markdown files
            )
        """
        os.makedirs(os.path.dirname(self.local_dir), exist_ok=True)
        
        if not os.path.exists(self.local_dir):
            # Clone repository if it doesn't exist locally
            try:
                print(f"Cloning repository {self.repo_url}...")
                Repo.clone_from(self.repo_url, self.local_dir, branch=self.branch)
            except GitCommandError as e:
                print(f"Error cloning repository: {str(e)}")
                return 0, 0, []
        else:
            # Repository exists, pull latest changes
            try:
                print(f"Pulling latest changes from {self.repo_url}...")
                repo = Repo(self.local_dir)
                origin = repo.remotes.origin
                origin.pull()
            except GitCommandError as e:
                print(f"Error pulling repository: {str(e)}")
                return 0, 0, []
        
        # Get the latest commit hash
        repo = Repo(self.local_dir)
        current_commit = repo.head.commit.hexsha
        
        # Check if already synced to this commit
        if self.tracking_data["last_commit"] == current_commit:
            print("Repository already up-to-date.")
            return 0, 0, []
        
        # Find all markdown files
        markdown_files = []
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
        
        # Check for new or modified files
        new_files = []
        changed_files = []
        updated_files = []
        
        for file_path in markdown_files:
            relative_path = os.path.relpath(file_path, self.local_dir)
            file_hash = self._calculate_file_hash(file_path)
            
            if relative_path not in self.tracking_data["files"]:
                new_files.append(file_path)
                updated_files.append(file_path)
            elif self.tracking_data["files"][relative_path] != file_hash:
                changed_files.append(file_path)
                updated_files.append(file_path)
            
            # Update tracking data
            self.tracking_data["files"][relative_path] = file_hash
        
        # Update tracking information
        self.tracking_data["last_commit"] = current_commit
        self.tracking_data["last_sync"] = datetime.now().isoformat()
        self._save_tracking_data()
        
        print(f"Found {len(new_files)} new and {len(changed_files)} changed markdown files")
        return len(new_files), len(changed_files), updated_files