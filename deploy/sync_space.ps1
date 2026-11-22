# Run from D:\panorama. Copies the app and the stitcher package into the Space clone and pushes.

if (-not (Select-String -Path deploy\app.py -Pattern "spaces.GPU" -Quiet)) {
    Write-Error "deploy\app.py has no @spaces.GPU placeholder; refusing to sync."
    exit 1
}
$space = "D:\space-repo"
Copy-Item deploy\app.py, deploy\requirements.txt, deploy\README.md $space -Force
Remove-Item "$space\stitcher" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item stitcher "$space\stitcher" -Recurse
Get-ChildItem "$space\stitcher" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Push-Location $space
git add -A
git commit -m "Sync from project repo"
git push
Pop-Location