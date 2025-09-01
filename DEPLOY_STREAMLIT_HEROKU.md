# Deploy — Streamlit & Heroku

## Streamlit Community Cloud
1. Push these files to a GitHub repo (include `streamlit_app.py`, `requirements_streamlit.txt`, `.streamlit/config.toml`, and `BostonHousing.csv`).
2. Go to https://share.streamlit.io → "New app" → select your repo/branch → set **Main file path** to `streamlit_app.py`.
3. In "Advanced settings", set **Python version** to match your local (e.g., 3.11).
4. First deploy will install packages from `requirements_streamlit.txt`.

## Heroku
1. Ensure you have Heroku CLI: `heroku login`.
2. Create an app: `heroku create dma818-boston-demo`.
3. Set buildpack to Python: `heroku buildpacks:set heroku/python`.
4. Commit files and push:
   ```bash
   git add streamlit_app.py requirements_streamlit.txt Procfile setup.sh BostonHousing.csv .streamlit/config.toml
   git commit -m "Deploy Streamlit app"
   git push heroku main
   ```
5. Open the app: `heroku open`.

> Procfile runs: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`


## Pull Request Previews (Review Apps via CI)
This repo includes two workflows:
- `deploy-preview.yml` — creates/updates a preview app on PR open/sync (`$HEROKU_APP_NAME-pr-<PR#>`).
- `cleanup-preview.yml` — deletes the preview app when the PR is closed.

**Required repo secrets** (same as main deploy):
- `HEROKU_API_KEY`
- `HEROKU_EMAIL`
- `HEROKU_APP_NAME` (base name; e.g., `dma818-boston-demo`).

> Note: App names must be globally unique in Heroku. If the base name collides, adjust your secret accordingly.
\n
## Enforce Required Status Checks
Run the **Enforce Branch Protection** workflow manually and set
- `branch` = main (or your default)
- `contexts` = comma-separated exact names from the Checks tab (e.g., `Python CI / build-test-run, Deploy Streamlit to Heroku / deploy`)

Store a repo secret `GH_TOKEN` with `repo` scope for the workflow.
