# Push to GitHub — eskgyimah/DMA818-BostonHousing

## Option A: Use the GitHub website (screenshot you shared)
1. Create a new repo at https://github.com/new
2. Owner: **eskgyimah**
3. Repository name: **DMA818-BostonHousing**
4. Keep it Public or Private, your call. Do **not** add README/.gitignore/license (we already have them).
5. Click **Create repository**.

## Option B: Push from local
```powershell
cd path\to\DMA818-BostonHousing
.\scripts\init_repo.ps1 -Remote "https://github.com/eskgyimah/DMA818-BostonHousing.git" -Branch main
```

or bash:
```bash
cd path/to/DMA818-BostonHousing
./scripts/init_repo.sh https://github.com/eskgyimah/DMA818-BostonHousing.git main
```

## After first push
- Add repo secrets (Settings → Secrets and variables → Actions):
  - HEROKU_API_KEY, HEROKU_APP_NAME, HEROKU_EMAIL, GH_TOKEN
- Open **Actions** tab → run **Enforce Branch Protection** (workflow_dispatch).
