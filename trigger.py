import sys
from streamlit import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", "interactive_scoring.py"]
    sys.exit(stcli.main())