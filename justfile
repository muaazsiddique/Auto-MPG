# List all recipes
default:
    @just --list

# Check dependencies
dependencies:
    @echo "Checking dependencies..."
    @(where uv >nul 2>nul && (echo "uv: INSTALLED - Version: " && uv --version)) || echo "uv: NOT INSTALLED"
    @(where just >nul 2>nul && (echo "just: INSTALLED - Version: " && just --version | findstr /R "just [0-9]")) || echo "just: NOT INSTALLED"
    @(where docker >nul 2>nul && (echo "docker: INSTALLED - Version: " && docker --version)) || echo "docker: NOT INSTALLED"
    @(where jq >nul 2>nul && (echo "jq: INSTALLED - Version: " && jq --version)) || echo "jq: NOT INSTALLED"
# Install dependencies in myenv
install:
    pip install numpy pandas matplotlib scikit-learn jupyter mlflow

# Run MLflow server
mlflow:
    @echo "MLflow server starting at: http://127.0.0.1:8080"
    mlflow server --host 127.0.0.1 --port 8080
# Run Jupyter notebook
notebook:
    jupyter notebook

# Create .env file
env:
    @powershell -Command "if (-not (Test-Path .env)) { \
        Write-Host 'Creating .env file...'; \
        Set-Content -Path '.env' -Value 'KERAS_BACKEND=tensorflow'; \
        Add-Content -Path '.env' -Value 'MLFLOW_TRACKING_URI=http://127.0.0.1:8080'; \
    } else { \
        Write-Host '.env file already exists.'; \
    }"