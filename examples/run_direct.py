"""
Direct runner for the FastHTML web interface.
"""
from finite_monkey.web.fasthtml_app import serve, app

if __name__ == "__main__":
    # This will directly call the serve function from fasthtml_app.py
    print("Starting Finite Monkey FastHTML web interface...")
    serve()