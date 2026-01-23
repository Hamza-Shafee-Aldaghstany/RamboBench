import argparse
import json

import javalang


# -------------------------------
# Clean generated output
# -------------------------------
def clean_output(output: str) -> str:
    cur_bracket = 0
    for idx, c in enumerate(output):
        if c == "{":
            cur_bracket += 1
        elif c == "}":
            cur_bracket -= 1

        if cur_bracket < 0:
            return output[:idx]

    return output


# -------------------------------
# Extract Java methods & types
# -------------------------------
def extract_methods_and_types(java_snippet: str):
    wrapped = f"""
    class Dummy {{
        void dummy() {{
            {java_snippet}
        }}
    }}
    """

    methods = set()
    types = set()

    try:
        tree = javalang.parse.parse(wrapped)
    except Exception:
        return [], []

    for _, node in tree:
        if isinstance(node, javalang.tree.MethodInvocation):
            methods.add(node.member)

        elif isinstance(node, javalang.tree.ClassCreator):
            types.add(node.type.name)

        elif isinstance(node, javalang.tree.VariableDeclaration):
            if hasattr(node.type, "name"):
                types.add(node.type.name)

    return sorted(methods), sorted(types)


# -------------------------------
# Main processing logic
# -------------------------------
def explode_predictions(input_path: str, output_path: str, input_col: str):
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for i, line in enumerate(fin):
            item = json.loads(line)

            predictions = item.get(input_col, [])
            for pred in predictions:
                clean_pred = clean_output(pred)
                methods, types = extract_methods_and_types(clean_pred)
                new_item = item.copy()
                new_item.pop(input_col, None)
                new_item.update(
                    {
                        f"{input_col}_id": i,
                        input_col: pred,
                        f"clean_{input_col}_code": clean_pred,
                        f"prediction_{input_col}_methods": methods,
                        f"prediction_{input_col}_types": types,
                    }
                )
                fout.write(json.dumps(new_item) + "\n")


# -------------------------------
# CLI entrypoint
# -------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Explode predictions_code into one JSONL row per prediction"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input JSONL file")
    parser.add_argument(
        "--output", "-o", required=True, help="Path to output JSONL file"
    )
    parser.add_argument("--input_col", required=True, help="name of the input coloumn")

    args = parser.parse_args()

    explode_predictions(args.input, args.output, args.input_col)

    print("✅ Explosion complete")
    print(f"📁 Saved to: {args.output}")


if __name__ == "__main__":
    main()
