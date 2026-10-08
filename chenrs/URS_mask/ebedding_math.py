import voyageai
import os
import re
import json
import numpy as np
from pathlib import Path

# Initialize Voyage AI client
vo = voyageai.Client(api_key="pa-iLfU04WvPBA2oZGULpUH-a1FqRp4VcevLPFwtZ3Ttsa")

# Directory containing problem representation files
repr_dir = Path(__file__).parent / "problem_representations"
output_file = Path(__file__).parent / "problem_repr_embeddings.json"
np_output_file = Path(__file__).parent / "problem_repr_embeddings.npy"
mapping_file = Path(__file__).parent / "problem_repr_embeddings_mapping.json"


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
    existing_reprs = set(embeddings_data.get("embeddings", {}).keys())

    print("\nScanning problem representation files...")
    repr_contents = {}

    # Get all txt files in problem_representations directory
    repr_files = sorted(repr_dir.glob("*.txt"))

    for repr_file in repr_files:
        problem_name = repr_file.stem  # e.g., 'vrpb.txt' -> 'vrpb'

        # Skip if already has embedding
        if problem_name in existing_reprs:
            print(f"  ⊙ {repr_file.name} (already cached)")
            continue

        print(f"Processing {repr_file.name}...")
        with open(repr_file, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content:
            repr_contents[problem_name] = content
            print(f"  ✓ Loaded content ({len(content)} characters)")
        else:
            print(f"  ✗ Empty file")

    # Check if there are new representations to embed
    if not repr_contents:
        print("\nNo new problem representations to embed!")
        print(f"All {len(existing_reprs)} representations already have embeddings.")
        return

    print(f"\nFound {len(repr_contents)} new problem representations to embed")
    print("Generating embeddings with Voyage AI...")

    # Prepare data for embedding
    problem_names = list(repr_contents.keys())
    problem_contents = list(repr_contents.values())

    result = vo.embed(problem_contents, model="voyage-code-3", output_dimension=256)

    # Add new embeddings to existing data
    for name, content, embedding in zip(
        problem_names, problem_contents, result.embeddings
    ):
        embeddings_data["embeddings"][name] = {
            "content": content,
            "embedding": embedding,
            "embedding_shape": len(embedding),
        }
        print(f"  ✓ {name}: embedding shape {len(embedding)}")

    # Save updated JSON
    print(f"\nSaving embeddings to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(embeddings_data, f, indent=2)

    # Update numpy array with all embeddings
    all_problem_names = sorted(embeddings_data["embeddings"].keys())
    all_embeddings = [
        embeddings_data["embeddings"][name]["embedding"] for name in all_problem_names
    ]
    embeddings_array = np.array(all_embeddings)
    np.save(np_output_file, embeddings_array)

    # Update mapping file
    mapping_data = {
        "problem_names": all_problem_names,
        "dimension": 256,
        "num_problems": len(all_problem_names),
    }
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, indent=2)

    print(f"✓ Saved embeddings to {output_file}")
    print(f"✓ Saved numpy array to {np_output_file}")
    print(f"✓ Saved mapping to {mapping_file}")
    print(f"\nSummary:")
    print(f"  - Previously cached: {len(existing_reprs)}")
    print(f"  - Newly embedded: {len(problem_names)}")
    print(f"  - Total problems: {len(all_problem_names)}")
    print(f"  - Embedding dimension: 256")
    print(f"  - New problems: {', '.join(problem_names)}")


if __name__ == "__main__":
    main()
