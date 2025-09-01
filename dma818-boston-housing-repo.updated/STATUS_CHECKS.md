# Status Checks — How to get exact names

GitHub branch protection requires **exact** status check context names.
For GitHub Actions, these are usually the **workflow name** or the **job name**.

## How to discover names
1. Open a completed PR or commit in GitHub → **Checks** tab.
2. Copy the green check row labels (e.g., `Python CI / build-test-run`, `Deploy Streamlit to Heroku / deploy`).
3. Use those labels verbatim in the `contexts` input of the `Enforce Branch Protection` workflow.

## Quick start
- Try contexts: `Python CI, Preview Deploy (Heroku)`
- If branch protection blocks merges due to missing checks, adjust contexts per the steps above.

## Automate via CLI (optional)
```bash
gh api repos/<owner>/<repo>/commits/<sha>/check-runs --paginate   | jq -r '.check_runs[].name' | sort -u
```
