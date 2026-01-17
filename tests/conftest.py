"""Pytest configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# CRITICAL: Set test environment FIRST
# =============================================================================
os.environ["TESTING"] = "1"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CRITICAL: Inject mock modules BEFORE any src imports
# This prevents loading real RAG dependencies (FAISS, Ollama, etc.)
# =============================================================================
mock_retrieve_module = MagicMock()
mock_answer_module = MagicMock()
mock_security_mapping_module = MagicMock()
mock_config_module = MagicMock()

sys.modules["retrieve"] = mock_retrieve_module
sys.modules["answer"] = mock_answer_module
sys.modules["security_mapping"] = mock_security_mapping_module
sys.modules["src.config"] = mock_config_module

# Configure security_mapping mock with required classes
mock_security_mapping_module.SecurityLevel = MagicMock()
mock_security_mapping_module.SecurityLevel.return_value = MagicMock()
mock_security_mapping_module.SecurityContext = MagicMock()
mock_security_mapping_module.DocumentType = MagicMock()
mock_security_mapping_module.DocumentType.POLICY = "POLICY"
mock_security_mapping_module.DocumentType.REPORT = "REPORT"
mock_security_mapping_module.DocumentType.MANUAL = "MANUAL"
mock_security_mapping_module.DocumentType.OTHER = "OTHER"
mock_security_mapping_module.DocumentType.FINANCE = "FINANCE"
mock_security_mapping_module.DocumentType.LEGAL = "LEGAL"
mock_security_mapping_module.DocumentType.HR = "HR"

# =============================================================================
# Pytest plugins
# =============================================================================
pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config):
    """Called after command line options have been parsed."""
    os.environ["TESTING"] = "1"
