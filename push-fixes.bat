@echo off
echo Pushing streamlit fixes to GitHub...
cd /d "%~dp0"
git add .
git commit --no-verify -m "fix: resolve duplicate widget keys and download labels"
git push
echo Done! Streamlit Cloud will redeploy automatically.
pause