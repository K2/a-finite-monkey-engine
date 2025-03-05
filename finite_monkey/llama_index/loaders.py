"""
Asynchronous document loaders for code files and GitHub issues
"""

import os
import re
import asyncio
import aiofiles
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import httpx
from llama_index.core.schema import Document
from llama_index.core.readers.base import BaseReader

logger = logging.getLogger(__name__)

class AsyncCodeLoader:
    """Asynchronous loader for code files"""
    
    async def load_data(
        self, 
        file_path: Optional[str] = None, 
        dir_path: Optional[str] = None,
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Load code files asynchronously
        
        Args:
            file_path: Path to a specific file to load
            dir_path: Path to a directory containing files to load
            recursive: Whether to search directories recursively
            extensions: List of file extensions to include (e.g., [".sol", ".py"])
            
        Returns:
            List of documents
        """
        # Set default file extensions
        if extensions is None:
            extensions = [".sol", ".js", ".ts", ".py", ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp"]
        
        # Load file or directory
        if file_path is not None:
            return await self._load_file(file_path)
        elif dir_path is not None:
            return await self._load_directory(dir_path, recursive, extensions)
        else:
            raise ValueError("Either file_path or dir_path must be provided")
    
    async def _load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file asynchronously
        
        Args:
            file_path: Path to the file
            
        Returns:
            List containing a single document
        """
        try:
            # Check if the file exists
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file asynchronously
            async with aiofiles.open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = await f.read()
            
            # Extract file metadata
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1]
            
            # Create a document
            metadata = {
                "file_path": file_path,
                "file_name": file_name,
                "file_type": file_ext.lstrip("."),
                "source_type": "file",
            }
            
            # Return the document
            return [Document(text=content, metadata=metadata)]
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            return []
    
    async def _load_directory(
        self, 
        dir_path: str, 
        recursive: bool,
        extensions: List[str],
    ) -> List[Document]:
        """
        Load files from a directory asynchronously
        
        Args:
            dir_path: Path to the directory
            recursive: Whether to search subdirectories
            extensions: List of file extensions to include
            
        Returns:
            List of documents
        """
        try:
            # Check if the directory exists
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            # Walk through the directory
            file_paths = []
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden directories if recursive
                if not recursive and root != dir_path:
                    continue
                
                # Add files with matching extensions
                for file_name in files:
                    file_ext = os.path.splitext(file_name)[1]
                    if file_ext in extensions:
                        file_path = os.path.join(root, file_name)
                        file_paths.append(file_path)
            
            # Load files concurrently
            tasks = [self._load_file(file_path) for file_path in file_paths]
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            return [doc for docs in results for doc in docs]
            
        except Exception as e:
            print(f"Error loading directory {dir_path}: {str(e)}")
            return []


class AsyncGithubIssueLoader(BaseReader):
    """
    A loader for loading GitHub issues asynchronously.
    
    This loader can load issues from a GitHub repository, including labels,
    comments, and other metadata. It supports authentication for private repositories.
    """
    
    def __init__(
        self,
        repo_url: str,
        token: Optional[str] = None,
        state: str = "all",
        labels: Optional[List[str]] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """
        Initialize the AsyncGithubIssueLoader.
        
        Args:
            repo_url: URL to the GitHub repository (e.g., "https://github.com/owner/repo")
            token: GitHub personal access token for authentication
            state: State of issues to load ("open", "closed", or "all")
            labels: List of label names to filter by
            since: ISO 8601 timestamp (e.g., "2020-01-01T00:00:00Z")
            limit: Maximum number of issues to load
        """
        super().__init__()
        self.repo_url = repo_url
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.state = state
        self.labels = labels
        self.since = since
        self.limit = limit
        
        # Parse GitHub URL
        regex = r"github\.com[:/]([^/]+)/([^/]+)"
        match = re.search(regex, self.repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub URL: {self.repo_url}")
        
        self.owner, self.repo = match.groups()
        self.repo = self.repo.rstrip(".git")
    
    async def load(self) -> List[Document]:
        """
        Load GitHub issues asynchronously.
        
        Returns:
            List of documents
        """
        # Construct API URL
        api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"
        
        # Prepare parameters
        params = {
            "state": self.state,
            "per_page": 100,
            "direction": "desc",
            "sort": "updated",
        }
        
        if self.labels:
            params["labels"] = ",".join(self.labels)
        
        if self.since:
            params["since"] = self.since
        
        # Prepare headers
        headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        
        if self.token:
            headers["Authorization"] = f"token {self.token}"
        
        # Fetch issues
        all_issues = []
        async with httpx.AsyncClient() as client:
            page = 1
            while True:
                params["page"] = page
                
                response = await client.get(api_url, params=params, headers=headers)
                if response.status_code != 200:
                    if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
                        raise ValueError("GitHub API rate limit exceeded")
                    else:
                        raise ValueError(f"Error fetching issues: {response.text}")
                
                issues = response.json()
                if not issues:
                    break
                
                all_issues.extend(issues)
                
                # Check if we've reached the limit
                if self.limit and len(all_issues) >= self.limit:
                    all_issues = all_issues[:self.limit]
                    break
                
                # Check if there are more pages
                link_header = response.headers.get("Link", "")
                if not re.search(r'rel="next"', link_header):
                    break
                
                page += 1
        
        # Process issues
        documents = []
        async with httpx.AsyncClient() as client:
            for issue in all_issues:
                # Skip pull requests
                if "pull_request" in issue:
                    continue
                
                # Fetch comments
                comments_url = issue["comments_url"]
                comments = []
                
                if issue["comments"] > 0:
                    response = await client.get(comments_url, headers=headers)
                    if response.status_code == 200:
                        comments = response.json()
                
                # Create document from issue
                doc = await self._create_issue_document(issue, comments)
                documents.append(doc)
        
        return documents
    
    async def _create_issue_document(
        self, issue: Dict[str, Any], comments: List[Dict[str, Any]]
    ) -> Document:
        """
        Create a Document from a GitHub issue.
        
        Args:
            issue: GitHub issue data
            comments: List of comments on the issue
        
        Returns:
            Document object
        """
        # Extract issue data
        issue_number = issue["number"]
        title = issue["title"]
        body = issue["body"] or ""
        user = issue["user"]["login"]
        created_at = issue["created_at"]
        updated_at = issue["updated_at"]
        state = issue["state"]
        labels = [label["name"] for label in issue["labels"]]
        
        # Build document text
        text = f"# {title}\n\n"
        text += f"Issue #{issue_number} - {state}\n"
        text += f"Created by {user} on {created_at}\n"
        text += f"Last updated: {updated_at}\n\n"
        
        if labels:
            text += f"Labels: {', '.join(labels)}\n\n"
        
        text += "## Description\n\n"
        text += body + "\n\n"
        
        # Add comments
        if comments:
            text += "## Comments\n\n"
            for comment in comments:
                comment_user = comment["user"]["login"]
                comment_date = comment["created_at"]
                comment_body = comment["body"] or ""
                
                text += f"### Comment by {comment_user} on {comment_date}\n\n"
                text += comment_body + "\n\n"
        
        # Create metadata
        metadata = {
            "source": f"{self.repo_url}/issues/{issue_number}",
            "issue_number": issue_number,
            "title": title,
            "user": user,
            "created_at": created_at,
            "updated_at": updated_at,
            "state": state,
            "labels": labels,
            "comments_count": len(comments),
            "type": "github_issue",
            "repository": f"{self.owner}/{self.repo}",
        }
        
        # Create document
        return Document(
            text=text,
            metadata=metadata,
            id_=f"github-issue-{self.owner}-{self.repo}-{issue_number}",
        )