"""
GitHub integration for the Finite Monkey web interface.

This module provides functionality for importing repositories from GitHub,
tracking branches and commits, and creating issues for discovered vulnerabilities.
"""

import os
import tempfile
import shutil
import asyncio
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel
import aiohttp

class GitHubBranch(BaseModel):
    """Model for GitHub branch information"""
    name: str
    commit_sha: str
    protected: bool = False

class GitHubCommit(BaseModel):
    """Model for GitHub commit information"""
    sha: str
    message: str
    author: str
    date: str
    url: str

class GitHubRepoInfo(BaseModel):
    """Model for GitHub repository information"""
    owner: str
    repo: str
    branches: List[GitHubBranch] = []
    commits: List[GitHubCommit] = []
    default_branch: str = "main"

class GitHubIssue(BaseModel):
    """Model for GitHub issue information"""
    title: str
    body: str
    labels: List[str] = ["security", "vulnerability"]

class GitHubClient:
    """Client for interacting with GitHub API"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub client.
        
        Args:
            token: Optional GitHub personal access token for authenticated requests
        """
        self.token = token
        self.headers = {}
        if token:
            self.headers["Authorization"] = f"token {token}"
        
    async def fetch_repo_info(self, repo_url: str) -> GitHubRepoInfo:
        """
        Fetch information about a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL (e.g. https://github.com/owner/repo)
            
        Returns:
            GitHubRepoInfo object containing repository information
        """
        # Extract owner and repo name from URL
        parts = repo_url.rstrip("/").split("/")
        owner = parts[-2]
        repo = parts[-1]
        
        # Create initial result
        result = GitHubRepoInfo(owner=owner, repo=repo)
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Get repository info to find default branch
            repo_api_url = f"https://api.github.com/repos/{owner}/{repo}"
            async with session.get(repo_api_url) as response:
                if response.status == 200:
                    repo_data = await response.json()
                    result.default_branch = repo_data.get("default_branch", "main")
                else:
                    # Raise exception if repository not found
                    error_data = await response.json()
                    raise ValueError(f"Error fetching repository: {error_data.get('message', 'Unknown error')}")
            
            # Get branches
            branches_url = f"{repo_api_url}/branches"
            async with session.get(branches_url) as response:
                if response.status == 200:
                    branches_data = await response.json()
                    result.branches = [
                        GitHubBranch(
                            name=branch["name"],
                            commit_sha=branch["commit"]["sha"],
                            protected=branch.get("protected", False)
                        )
                        for branch in branches_data
                    ]
            
            # Get recent commits (default branch)
            commits_url = f"{repo_api_url}/commits?sha={result.default_branch}"
            async with session.get(commits_url) as response:
                if response.status == 200:
                    commits_data = await response.json()
                    result.commits = [
                        GitHubCommit(
                            sha=commit["sha"],
                            message=commit["commit"]["message"],
                            author=commit["commit"]["author"]["name"],
                            date=commit["commit"]["author"]["date"],
                            url=commit["html_url"]
                        )
                        for commit in commits_data[:10]  # Limit to 10 most recent commits
                    ]
        
        return result
    
    async def clone_repository(self, repo_url: str, branch: str = "main", commit_sha: Optional[str] = None) -> str:
        """
        Clone a GitHub repository to a temporary directory.
        
        Args:
            repo_url: GitHub repository URL
            branch: Branch to clone
            commit_sha: Optional commit SHA to checkout
            
        Returns:
            Path to the cloned repository
        """
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="finite_monkey_github_")
        
        # Run git clone command
        proc = await asyncio.create_subprocess_exec(
            "git", "clone", "--branch", branch, repo_url, temp_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            # Clean up and raise exception
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise ValueError(f"Failed to clone repository: {stderr.decode()}")
        
        # Checkout specific commit if requested
        if commit_sha:
            proc = await asyncio.create_subprocess_exec(
                "git", "-C", temp_dir, "checkout", commit_sha,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            _, stderr = await proc.communicate()
            
            if proc.returncode != 0:
                # Clean up and raise exception
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise ValueError(f"Failed to checkout commit: {stderr.decode()}")
        
        return temp_dir
    
    async def create_issue(self, owner: str, repo: str, issue: GitHubIssue) -> Dict[str, Any]:
        """
        Create a new issue in a GitHub repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            issue: Issue data
            
        Returns:
            Created issue data from GitHub API
        """
        if not self.token:
            raise ValueError("GitHub token is required to create issues")
        
        async with aiohttp.ClientSession(headers=self.headers) as session:
            issues_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            payload = {
                "title": issue.title,
                "body": issue.body,
                "labels": issue.labels
            }
            
            async with session.post(issues_url, json=payload) as response:
                if response.status in (201, 200):
                    return await response.json()
                else:
                    error_data = await response.json()
                    raise ValueError(f"Error creating issue: {error_data.get('message', 'Unknown error')}")