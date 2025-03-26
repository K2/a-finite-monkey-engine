import streamlit.web.cli as stcli
import sys
from pathlib import Path

if __name__ == "__main__":
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Run Streamlit app
    app_path = project_root / "finite_monkey" / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())
