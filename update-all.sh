#!/bin/bash
set -e

###############################################
# CONFIG
###############################################
REQUIRED_PY_VERSION="3.12"
###############################################

echo "🚀 Starting dependency update process using Python $REQUIRED_PY_VERSION..."

# Detect interpreter name
CANDIDATES=(
  "python$REQUIRED_PY_VERSION"
  "python${REQUIRED_PY_VERSION//./}"
  "python3"
  "python"
)

# Windows-only candidate (py launcher)
if command -v py >/dev/null 2>&1; then
  CANDIDATES+=("py -$REQUIRED_PY_VERSION")
fi

PY_INTERPRETER=""

echo "🔍 Searching for Python $REQUIRED_PY_VERSION interpreter..."

for C in "${CANDIDATES[@]}"; do
  if $C -V >/dev/null 2>&1; then
    # Check exact version
    V=$($C -V 2>&1 | awk '{print $2}')
    if [[ "$V" == "$REQUIRED_PY_VERSION"* ]]; then
      PY_INTERPRETER="$C"
      break
    fi
  fi
done

if [ -z "$PY_INTERPRETER" ]; then
  echo "❌ Python $REQUIRED_PY_VERSION was not found on this system."
  echo "Please install it and try again."
  exit 1
fi

echo "✅ Found interpreter: $PY_INTERPRETER ($( $PY_INTERPRETER -V ))"

# Step 1: Validate input
if [ ! -f requirements.txt ]; then
  echo "❌ requirements.txt not found. Aborting."
  exit 1
fi

# Step 2: Extract top-level names
echo "🧹 Generating requirements.in from requirements.txt..."
cut -d '=' -f 1 requirements.txt | sed 's/ *$//' | sort -u > requirements.in

# Step 3: Remove old venv
echo "♻️ Removing old virtual environment (.venv)..."
rm -rf .venv

# Step 4: Create fresh Python venv
echo "📦 Creating new Python $REQUIRED_PY_VERSION virtual environment..."
$PY_INTERPRETER -m venv .venv

# Step 5: Activate venv
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
  source .venv/Scripts/activate
else
  source .venv/bin/activate
fi

echo "🐍 Python version inside venv: $(python -V)"

# Step 6: Install pip-tools
echo "🛠 Installing pip-tools..."
python -m pip install --upgrade "pip<24.3" setuptools wheel
python -m pip install "pip-tools>=7.4,<7.6"

# Step 7: pip-compile upgrade
echo "📡 Upgrading dependencies using pip-compile..."
pip-compile --upgrade requirements.in --output-file requirements.txt

# Step 8: Install updated requirements
echo "📥 Installing updated packages into .venv..."
pip install -r requirements.txt

# Step 9: Keep only top-level pins
echo "🧾 Cleaning requirements.txt to include only top-level packages..."
pip list --not-required --format=freeze | sort > requirements.txt

echo "✅ All dependencies updated using Python $REQUIRED_PY_VERSION!"
echo "📂 Virtual environment: .venv/"
echo "📌 Clean pinned packages: requirements.txt"
echo "🧾 Top-level names: requirements.in"
