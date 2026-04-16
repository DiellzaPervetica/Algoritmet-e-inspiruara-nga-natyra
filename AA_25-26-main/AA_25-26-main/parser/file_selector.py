from pathlib import Path


def select_file(input_dir=None):
    project_root = Path(__file__).resolve().parents[1]
    resolved_input_dir = Path(input_dir) if input_dir else project_root / "data" / "input"

    if not resolved_input_dir.is_absolute():
        resolved_input_dir = project_root / resolved_input_dir

    if not resolved_input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {resolved_input_dir}")

    files = sorted(f.name for f in resolved_input_dir.iterdir() if f.suffix == ".json")

    if not files:
        raise FileNotFoundError(f"No JSON files found in {resolved_input_dir}")

    print("Available files:")
    for idx, file in enumerate(files):
        print(f"{idx}: {file}")

    while True:
        try:
            choice = int(input("Select a file by index: "))
            if 0 <= choice < len(files):
                break
            else:
                print("Invalid index, try again.")
        except ValueError:
            print("Please enter a valid number.")

    return str(resolved_input_dir / files[choice])
