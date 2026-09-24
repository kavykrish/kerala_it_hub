import os
import sys

# Guarantees the project root is importable as `web_retrieval.*`,
# `backend.*`, `mcp_server.*` etc. regardless of where pytest is
# invoked from -- matches the sys.path.insert pattern the existing
# tests/day*_test.py scripts already use individually.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
