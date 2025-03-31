#!/usr/bin/env python3
"""
Utility to populate the threat vector database with known vulnerability patterns.
This script indexes common security vulnerability patterns and their associated metadata.
"""
import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any
from pathlib import Path
from loguru import logger

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from tools.vector_store_util import SimpleVectorStore

# Default threat patterns to use if no file specified
DEFAULT_THREAT_PATTERNS = [
    {
        "text": "username = request.POST['username']\npassword = request.POST['password']\nquery = f\"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'\"",
        "metadata": {
            "threat_type": "SQL Injection",
            "severity": "high",
            "language": "python",
            "description": "String formatting used in SQL query with unsanitized user input",
            "cwe": "CWE-89"
        }
    },
    {
        "text": "const userInput = req.body.userInput;\nconst query = 'SELECT * FROM users WHERE id = ' + userInput;\ndb.query(query, (err, results) => {\n  // ...\n});",
        "metadata": {
            "threat_type": "SQL Injection",
            "severity": "high",
            "language": "javascript",
            "description": "Direct concatenation of user input in SQL query",
            "cwe": "CWE-89"
        }
    },
    {
        "text": "command = request.GET.get('command', '')\nos.system(command)",
        "metadata": {
            "threat_type": "Command Injection",
            "severity": "high",
            "language": "python",
            "description": "Unsanitized user input passed to os.system",
            "cwe": "CWE-78"
        }
    },
    {
        "text": "exec(\"rm -rf \" + req.query.dir)",
        "metadata": {
            "threat_type": "Command Injection",
            "severity": "high",
            "language": "javascript",
            "description": "Unsanitized user input in exec command",
            "cwe": "CWE-78"
        }
    },
    {
        "text": "filename = request.GET.get('filename')\nwith open(filename, 'r') as f:\n    content = f.read()",
        "metadata": {
            "threat_type": "Path Traversal",
            "severity": "high",
            "language": "python",
            "description": "Unsanitized user input used in file path",
            "cwe": "CWE-22"
        }
    },
    {
        "text": "const userFile = req.query.file;\nconst data = fs.readFileSync(userFile);",
        "metadata": {
            "threat_type": "Path Traversal",
            "severity": "high",
            "language": "javascript",
            "description": "User-controlled file path in file system operation",
            "cwe": "CWE-22"
        }
    },
    {
        "text": "user_data = request.POST.get('data')\ndeserialized_data = pickle.loads(user_data)",
        "metadata": {
            "threat_type": "Insecure Deserialization",
            "severity": "high",
            "language": "python",
            "description": "Deserializing user-supplied pickle data, which can lead to RCE",
            "cwe": "CWE-502"
        }
    },
    {
        "text": "user_html = request.GET.get('html', '')\nresponse.write(f'<div>{user_html}</div>')",
        "metadata": {
            "threat_type": "Cross-Site Scripting (XSS)",
            "severity": "medium",
            "language": "python",
            "description": "Unsanitized user input included in HTML response",
            "cwe": "CWE-79"
        }
    },
    {
        "text": "document.getElementById('output').innerHTML = userInput;",
        "metadata": {
            "threat_type": "Cross-Site Scripting (XSS)",
            "severity": "medium",
            "language": "javascript",
            "description": "Unsanitized user input assigned to innerHTML",
            "cwe": "CWE-79"
        }
    },
    {
        "text": "password = 'hardcoded_secret'",
        "metadata": {
            "threat_type": "Hardcoded Credentials",
            "severity": "medium",
            "language": "python",
            "description": "Hardcoded password in source code",
            "cwe": "CWE-798"
        }
    },
    {
        "text": "const apiKey = \"1234567890abcdef\";",
        "metadata": {
            "threat_type": "Hardcoded Credentials",
            "severity": "medium",
            "language": "javascript",
            "description": "Hardcoded API key in source code",
            "cwe": "CWE-798"
        }
    },
    {
        "text": "app.use(bodyParser.json({verify: false}));",
        "metadata": {
            "threat_type": "Missing Validation",
            "severity": "medium",
            "language": "javascript",
            "description": "JSON parser with verification disabled",
            "cwe": "CWE-20"
        }
    }
]

async def load_threat_patterns(file_path: str = None) -> List[Dict[str, Any]]:
    """
    Load threat patterns from a file or use defaults.
    
    Args:
        file_path: Path to JSON file with threat patterns, or None to use defaults
        
    Returns:
        List of threat patterns
    """
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            logger.info(f"Loaded {len(patterns)} threat patterns from {file_path}")
            return patterns
        except Exception as e:
            logger.error(f"Error loading threat patterns from {file_path}: {e}")
    
    logger.info(f"Using {len(DEFAULT_THREAT_PATTERNS)} default threat patterns")
    return DEFAULT_THREAT_PATTERNS

async def populate_threat_database(
    patterns: List[Dict[str, Any]],
    vector_store_dir: str = "./vector_store",
    collection_name: str = "threats",
    embedding_model: str = "local",
    embedding_device: str = "auto"
) -> bool:
    """
    Populate the threat vector database with vulnerability patterns.
    
    Args:
        patterns: List of threat patterns to add
        vector_store_dir: Directory for vector store
        collection_name: Collection name for threat patterns
        embedding_model: Embedding model to use
        embedding_device: Device to run embeddings on
        
    Returns:
        Success status
    """
    try:
        # Initialize vector store
        vector_store = SimpleVectorStore(
            storage_dir=vector_store_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=embedding_device
        )
        
        # Add threat patterns to vector store
        success = await vector_store.add_documents(patterns, show_progress=True)
        
        if success:
            logger.info(f"Successfully added {len(patterns)} threat patterns to vector store")
        else:
            logger.error("Failed to add threat patterns to vector store")
        
        return success
    except Exception as e:
        logger.error(f"Error populating threat database: {e}")
        return False

async def test_threat_detection(
    vector_store_dir: str = "./vector_store",
    collection_name: str = "threats",
    embedding_model: str = "local",
    embedding_device: str = "auto"
) -> None:
    """
    Test threat detection by querying a few sample code snippets.
    
    Args:
        vector_store_dir: Directory for vector store
        collection_name: Collection name for threat patterns
        embedding_model: Embedding model to use
        embedding_device: Device to run embeddings on
    """
    try:
        # Initialize vector store
        vector_store = SimpleVectorStore(
            storage_dir=vector_store_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            embedding_device=embedding_device
        )
        
        # Sample vulnerable code snippets to test
        test_samples = [
            "username = request.form.get('username')\npassword = request.form.get('password')\ncursor.execute(f\"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'\")",
            "app.get('/user', (req, res) => {\n  const userId = req.query.id;\n  const userQuery = `SELECT * FROM users WHERE id = ${userId}`;\n  db.query(userQuery, (err, result) => {\n    res.json(result);\n  });\n});",
            "import pickle\nuser_data = request.cookies.get('session')\nif user_data:\n    user = pickle.loads(base64.b64decode(user_data))"
        ]
        
        # Test each sample
        for i, sample in enumerate(test_samples):
            logger.info(f"Testing sample {i+1}:")
            logger.info("-" * 40)
            logger.info(sample)
            logger.info("-" * 40)
            
            # Query with the sample
            results = await vector_store.query_with_adaptive_prompts(
                query_text=sample,
                min_k=3,
                max_k=10,
                similarity_threshold=0.5,
                drop_off_factor=0.3
            )
            
            # Display results
            for j, result in enumerate(results.get('results', [])[:3]):
                metadata = result.get('metadata', {})
                logger.info(f"Match {j+1}: {metadata.get('threat_type', 'Unknown')} "
                           f"(Score: {result.get('score', 0):.4f}, Severity: {metadata.get('severity', 'unknown')})")
                logger.info(f"Description: {metadata.get('description', 'No description')}")
                logger.info("-" * 40)
            
            logger.info("\n")
    
    except Exception as e:
        logger.error(f"Error testing threat detection: {e}")

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Populate and test the threat vector database')
    parser.add_argument('-f', '--file', help='JSON file with threat patterns (optional)')
    parser.add_argument('-d', '--dir', default='./vector_store', help='Vector store directory')
    parser.add_argument('-c', '--collection', default='threats', help='Collection name for threats')
    parser.add_argument('-m', '--model', default='local', choices=['local', 'ipex', 'ollama'], 
                       help='Embedding model')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'xpu'],
                       help='Device to run embeddings on')
    parser.add_argument('-t', '--test', action='store_true', help='Run threat detection test after populating')
    parser.add_argument('--only-test', action='store_true', help='Only run the threat detection test')
    
    args = parser.parse_args()
    
    # Configure logger
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    if not args.only_test:
        # Load threat patterns
        patterns = await load_threat_patterns(args.file)
        
        # Populate threat database
        success = await populate_threat_database(
            patterns=patterns,
            vector_store_dir=args.dir,
            collection_name=args.collection,
            embedding_model=args.model,
            embedding_device=args.device
        )
        
        if not success:
            logger.error("Failed to populate threat database")
            return 1
    
    # Run test if requested
    if args.test or args.only_test:
        await test_threat_detection(
            vector_store_dir=args.dir,
            collection_name=args.collection,
            embedding_model=args.model,
            embedding_device=args.device
        )
    
    return 0

if __name__ == "__main__":
    asyncio.run(main())
