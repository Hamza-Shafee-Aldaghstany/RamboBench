import argparse
import json
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--merged_file", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    files = sorted(
        [
            f
            for f in os.listdir(args.output_dir)
            if f.startswith("predictions_") and f.endswith(".jsonl")
        ]
    )

    with open(args.merged_file, "w") as fout:
        for fn in files:
            path = os.path.join(args.output_dir, fn)
            with open(path, "r") as f:
                for line in f:
                    fout.write(line)

    print(f"Merged {len(files)} files into {args.merged_file}")


if __name__ == "__main__":
    main()
