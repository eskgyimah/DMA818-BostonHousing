@echo off
echo Starting local Streamlit server...
cd /d "%~dp0"
echo Server will be available at: http://localhost:8504
python -m streamlit run streamlit_app.py --server.port 8504
pause