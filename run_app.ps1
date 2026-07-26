$projectRoot = $PSScriptRoot

# Start Backend Server
Write-Host "Starting FastAPI Backend Server..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; .\venv\Scripts\Activate.ps1; cd backend; uvicorn app:app --reload --port 8000"

# Wait a moment for backend to initialize
Start-Sleep -Seconds 2

# Start React Frontend UI
Write-Host "Starting React Frontend UI..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\frontend'; npm run dev"

Write-Host "Launched Backend and Frontend servers successfully." -ForegroundColor Cyan
