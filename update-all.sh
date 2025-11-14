#!/bin/bash
set -e

echo "🚀 Starting dependency update process..."

# Step 1: Validate input
if [ ! -f requirements.txt ]; then
  echo "❌ requirements.txt not found. Aborting."
  exit 1
fi

# Step 2: Extract top-level package names (no versions) to requirements.in
echo "🧹 Generating requirements.in from requirements.txt..."
cut -d '=' -f 1 requirements.txt | sed 's/ *$//' | sort -u > requirements.in

# Step 3: Remove old venv
echo "♻️ Removing old virtual environment (.venv)..."
rm -rf .venv

# Step 4: Create fresh virtual environment
echo "📦 Creating new virtual environment..."
python -m venv .venv

# Step 5: Activate venv (cross-platform)
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

# Step 6: Upgrade pip and install pip-tools
echo "🛠 Installing pip-tools..."
python -m pip install --upgrade "pip<24.3" setuptools wheel
python -m pip install "pip-tools>=7.4,<7.6"

# Step 7: Use pip-compile to get the latest compatible versions
echo "📡 Upgrading dependencies using pip-compile..."
pip-compile --upgrade requirements.in --output-file requirements.txt

# Step 8: Install updated requirements
echo "📥 Installing updated packages into .venv..."
pip install -r requirements.txt

# Step 9: Regenerate minimal top-level pinned requirements.txt
echo "🧾 Cleaning requirements.txt to include only top-level packages..."
pip list --not-required --format=freeze | sort > requirements.txt

echo "✅ All dependencies updated!"
echo "📂 Virtual environment: .venv/"
echo "📌 Clean pinned top-level packages: requirements.txt"
echo "🧾 Editable top-level names: requirements.in"
