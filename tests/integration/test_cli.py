"""Quick CLI test - Verify all commands."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list) -> str:
    """Run CLI command and return output."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )
    return result.stdout + result.stderr


def test_cli():
    """Test CLI commands."""
    print("🧪 Testing CLI Commands\n" + "=" * 60)

    # Test 1: Main help
    print("\n1️⃣ Testing: python -m cli.main --help")
    output = run_command([sys.executable, "-m", "cli.main", "--help"])
    assert "Stock Prediction ML Pipeline CLI" in output
    print("✅ Main help working")

    # Test 2: Train help
    print("\n2️⃣ Testing: python -m cli.main train --help")
    output = run_command([sys.executable, "-m", "cli.main", "train", "--help"])
    assert "--ticker" in output
    assert "--epochs" in output
    print("✅ Train command working")

    # Test 3: Predict help
    print("\n3️⃣ Testing: python -m cli.main predict --help")
    output = run_command([sys.executable, "-m", "cli.main", "predict", "--help"])
    assert "--model-path" in output
    assert "--ticker" in output  # Comando simplificado: prediz apenas próximo dia
    print("✅ Predict command working")

    # Test 4: Evaluate help
    print("\n4️⃣ Testing: python -m cli.main evaluate --help")
    output = run_command([sys.executable, "-m", "cli.main", "evaluate", "--help"])
    assert "--model-path" in output
    print("✅ Evaluate command working")

    # Test 5: Monitor help
    print("\n5️⃣ Testing: python -m cli.main monitor --help")
    output = run_command([sys.executable, "-m", "cli.main", "monitor", "--help"])
    assert "--port" in output
    print("✅ Monitor command working")

    print("\n" + "=" * 60)
    print("🎉 All CLI tests passed!")


if __name__ == "__main__":
    test_cli()
