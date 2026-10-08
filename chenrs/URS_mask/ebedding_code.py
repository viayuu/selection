import voyageai
import os
import re
import json
import numpy as np
from pathlib import Path

# Initialize Voyage AI client
vo = voyageai.Client(api_key="pa-iLfU04WvPBA2oZGULpUH-a1FqRp4VcevLPFwtZ3Ttsa")

# Directory containing mask files
mask_dir = Path(__file__).parent / "masks_unified"
output_file = Path(__file__).parent / "mask_embeddings.json"
np_output_file = Path(__file__).parent / "mask_embeddings.npy"
mapping_file = Path(__file__).parent / "mask_embeddings_mapping.json"


def extract_mask_function(file_path):
    """Extract mask function code from def to return"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Try multiple patterns to extract the mask function
    # Pattern 1: Match from def to return statement (most common case)
    pattern1 = r"(def\s+mask_\w+\s*\([^)]*\):.*?return\s+\w+)"
    matches = re.findall(pattern1, content, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Pattern 2: Match entire function block (fallback)
    pattern2 = r"(def\s+mask_\w+\s*\([^)]*\):.*?)(?=\n\s*\n|\n[^\s\t]|\Z)"
    matches = re.findall(pattern2, content, re.DOTALL)

    if matches:
        func_code = matches[0].strip()
        # Make sure it has a return statement
        if "return" in func_code:
            return func_code

    return None


def load_existing_embeddings():
    """Load existing embeddings from cache file"""
    if output_file.exists():
        print(f"Loading existing embeddings from {output_file}...")
        with open(output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  ✓ Loaded {len(data.get('embeddings', {}))} existing embeddings")
        return data
    else:
        print("No existing embeddings found, starting fresh...")
        return {"model": "voyage-code-3", "dimension": 256, "embeddings": {}}


def main():
    # Load existing embeddings first
    embeddings_data = load_existing_embeddings()
    existing_masks = set(embeddings_data.get("embeddings", {}).keys())

    print("\nScanning mask files...")
    mask_functions = {}

    # Get all Python mask files, excluding mask_registry.py
    mask_files = sorted(
        [f for f in mask_dir.glob("mask_*.py") if f.name != "mask_registry.py"]
    )

    for mask_file in mask_files:
        problem_name = mask_file.stem  # e.g., 'mask_vrpb' -> 'mask_vrpb'

        # Skip if already has embedding
        if problem_name in existing_masks:
            print(f"  ⊙ {mask_file.name} (already cached)")
            continue

        print(f"Processing {mask_file.name}...")
        func_code = extract_mask_function(mask_file)
        if func_code:
            mask_functions[problem_name] = func_code
            print(f"  ✓ Extracted function ({len(func_code)} characters)")
        else:
            print(f"  ✗ Could not extract function")

    # Check if there are new masks to embed
    if not mask_functions:
        print("\nNo new mask functions to embed!")
        print(f"All {len(existing_masks)} masks already have embeddings.")
        return

    print(f"\nFound {len(mask_functions)} new mask functions to embed")
    print("Generating embeddings with Voyage AI...")

    # Prepare data for embedding
    function_names = list(mask_functions.keys())
    function_codes = list(mask_functions.values())

    # Get embeddings from Voyage AI
    result = vo.embed(function_codes, model="voyage-code-3", output_dimension=256)

    # Add new embeddings to existing data
    for name, code, embedding in zip(function_names, function_codes, result.embeddings):
        embeddings_data["embeddings"][name] = {
            "code": code,
            "embedding": embedding,
            "embedding_shape": len(embedding),
        }
        print(f"  ✓ {name}: embedding shape {len(embedding)}")

    # Save updated JSON
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=2)

    # Update numpy array with all embeddings
    all_function_names = sorted(embeddings_data["embeddings"].keys())
    all_embeddings = [
        embeddings_data["embeddings"][name]["embedding"] for name in all_function_names
    ]
    embeddings_array = np.array(all_embeddings)
    np.save(np_output_file, embeddings_array)

    # Update mapping file
    mapping_data = {
        "function_names": all_function_names,
        "dimension": 256,
        "num_functions": len(all_function_names),
    }
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    print(f"✓ Saved embeddings to {output_file}")
    print(f"✓ Saved numpy array to {np_output_file}")
    print(f"✓ Saved mapping to {mapping_file}")
    print(f"\nSummary:")
    print(f"  - Previously cached: {len(existing_masks)}")
    print(f"  - Newly embedded: {len(function_names)}")
    print(f"  - Total functions: {len(all_function_names)}")
    print(f"  - Embedding dimension: 256")
    print(
        f"  - New functions: {', '.join([name.replace('mask_', '') for name in function_names])}"
    )


if __name__ == "__main__":
    main()
