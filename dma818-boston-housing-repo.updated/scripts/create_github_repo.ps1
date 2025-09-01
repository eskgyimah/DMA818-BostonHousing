Param(
  [string]$Name = "DMA818-BostonHousing",
  [string]$Private = "true"
)
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Error "GitHub CLI (gh) not found. See https://cli.github.com/"
  exit 1
}
$privacy = if ($Private -eq "true") { "--private" } else { "--public" }
gh repo create $Name $privacy --source "." --remote "origin" --push
