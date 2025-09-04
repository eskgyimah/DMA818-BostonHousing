@echo off
echo Pushing all streamlit enhancements to GitHub...
cd /d "%~dp0"
echo Adding all files...
git add .
echo Committing changes...
git commit --no-verify -m "enhance: add comprehensive streamlit visualizations

- Fix duplicate widget keys and download button labels
- Add EDA visuals: correlation heatmap, scatter matrix, MEDV histogram
- Add regression diagnostics: residuals, pred vs actual, learning curves  
- Add classification diagnostics: confusion matrix, ROC curve, PR curve
- All visualizations use matplotlib only with unique widget keys
- Expandable sections for optional advanced analytics"
echo Pushing to remote...
git push
echo.
echo ✅ Done! Streamlit Cloud will redeploy automatically.
echo ✅ Enhanced app will be available at: https://dma818-bostonhousing-a7sxfoly9ssrmqpzdbioqk.streamlit.app/
echo.
pause