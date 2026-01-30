#!/usr/bin/env python3
"""
Parallelized version of your d4j eval harness using ProcessPoolExecutor.
Usage (example):
    python d4j_eval_harness_bugmap_list.py --pred preds.jsonl --out results.jsonl --workers 6 --max 50
"""

# ---------------------------
# (unchanged constants/helpers omitted in this snippet for brevity)
# Paste your PROJECT_MAP, PROJECT_BUG_LIST, DEFECTS4J_BIN, env, and all helper functions here:
# - parse_project_bug_list
# - load_bug_map_file
# - build_project_bug_map
# - normalize_prediction_body
# - rel_path_from_fpath_tuple
# - read_function_by_name_from_file
# - replace_method_body_using_brace_count
# (Make sure these remain at module top-level so worker processes can import them.)
# ---------------------------
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import javalang
from tqdm import tqdm

# ---------------------------
# Configure project -> defects4j project id (project name used by defects4j)
# ---------------------------
PROJECT_MAP = {
    "apache_commons-codec": "Codec",
    "apache_commons-lang": "Lang",
    "apache_commons-math": "Math",
    "google_closure-compiler": "Closure",
    "jfree_jfreechart": "Chart",
    "apache_commons-collections": "Collections",
    "apache_commons-compress": "Compress",
    "apache_commons-jxpath": "JxPath",
    "apache_commons-csv": "Csv",
    "FasterXML_jackson-databind": "JacksonDatabind",
    "FasterXML_jackson-core": "JacksonCore",
    "google_gson": "Gson",
    "jhy_jsoup": "Jsoup",
    "JodaOrg_joda-time": "Time",
}

# ---------------------------
# Option 1: Provide bug mapping as a LIST here (one entry per project).
# Entry format: "project_prefix:bug_id"
# Example:
PROJECT_BUG_LIST = [
    "JodaOrg_joda-time:1",
    "FasterXML_jackson-databind:112",
    "jhy_jsoup:93",
    "apache_commons-csv:16",
    "jfree_jfreechart:1",
    "apache_commons-codec:18",
    "apache_commons-lang:1",
    "FasterXML_jackson-core:26",
    "apache_commons-jxpath:22",
    "google_gson:18",
    "apache_commons-compress:47",
    "apache_commons-collections:28",
    "apache_commons-math:1",
    "google_closure-compiler:107",
]
DEFECTS4J_BIN = "/home/user/Desktop/Code_Bench/ujb/defects4j/framework/bin/defects4j"

# Make sure environment has DEFECTS4J_HOME
env = os.environ.copy()
env["DEFECTS4J_HOME"] = "/home/user/Desktop/Code_Bench/ujb/defects4j"


# ---------------------------
# Helpers to build mapping from list or file
# ---------------------------
def parse_project_bug_list(entries: List[str]) -> Dict[str, str]:
    """
    Parse list entries like "project_prefix:bug_id" into dict.
    Accepts entries that contain ':' or whitespace separators.
    """
    result = {}
    for e in entries:
        if not e:
            continue
        if isinstance(e, dict):
            # allow dict entries directly
            for k, v in e.items():
                result[str(k)] = str(v)
            continue
        s = str(e).strip()
        # try JSON dict-ish string
        if s.startswith("{") and s.endswith("}"):
            try:
                d = json.loads(s)
                if isinstance(d, dict):
                    for k, v in d.items():
                        result[str(k)] = str(v)
                    continue
            except Exception:
                pass
        # try "project:bug" first
        if ":" in s:
            left, right = s.split(":", 1)
            result[left.strip()] = right.strip()
            continue
        # try whitespace separated
        parts = s.split()
        if len(parts) >= 2:
            result[parts[0].strip()] = parts[1].strip()
            continue
        # if we can't parse, raise to make error visible
        raise ValueError(f"Unrecognized PROJECT_BUG_LIST entry format: '{s}'")
    return result


def load_bug_map_file(path: str) -> Dict[str, str]:
    """
    Load bug map from a file. Supports:
      - JSON object: {"project_prefix": "66", ...}
      - newline-separated list of "project_prefix:bug_id" (or whitespace-separated)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    # try JSON
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
        # if JSON array, try parse as list entries
        if isinstance(obj, list):
            return parse_project_bug_list([str(x) for x in obj])
    except Exception:
        pass
    # fallback: parse as newline-separated list
    lines = [
        ln.strip()
        for ln in txt.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    return parse_project_bug_list(lines)


def build_project_bug_map(cmdline_bug_map_path: str = None) -> Dict[str, str]:
    """
    Build final project -> bug_id mapping from:
      1) PROJECT_BUG_LIST in this script
      2) optionally augmented by file provided with --bug-map
    """
    mapping = {}
    # parse in-script list first
    if PROJECT_BUG_LIST:
        mapping.update(parse_project_bug_list(PROJECT_BUG_LIST))
    # then override/extend with file if provided
    if cmdline_bug_map_path:
        file_map = load_bug_map_file(cmdline_bug_map_path)
        mapping.update(file_map)
    return mapping


# ---------------------------
# Reused helpers (method locate/replace/test)
# ---------------------------
def normalize_prediction_body(pred: str) -> str:
    p = pred.strip()
    # if p.startswith("{") and p.endswith("}"):
    #     p = p[1:-1].strip()
    return p


def rel_path_from_fpath_tuple(fpath_tuple: List[str]) -> str:
    if not fpath_tuple or len(fpath_tuple) < 3:
        return os.path.join(*fpath_tuple)
    return os.path.join(*fpath_tuple[2:])


def read_function_by_name_from_file(
    full_path: str, class_name: str, function_name: str
) -> Tuple[str, int, int]:
    if not os.path.exists(full_path):
        raise FileNotFoundError(full_path)
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()
    try:
        tree = javalang.parse.parse(source)
    except Exception as e:
        raise RuntimeError(f"Java parse failed: {e}")
    lines = source.splitlines()
    for _, node in tree.filter(javalang.tree.ClassDeclaration):
        if node.name != class_name:
            continue
        for method in node.methods:
            if method.name != function_name:
                continue
            if method.position is None:
                raise RuntimeError("Method position missing")
            start_line = method.position.line - 1
            brace_count = 0
            started = False
            extracted = []
            end_line = start_line
            for idx, line in enumerate(lines[start_line:], start=start_line):
                extracted.append(line)
                brace_count += line.count("{")
                brace_count -= line.count("}")
                if "{" in line:
                    started = True
                if started and brace_count == 0:
                    end_line = idx
                    break
            return "\n".join(extracted), start_line, end_line
    raise ValueError(f"{class_name}.{function_name} not found")


def replace_method_body_using_brace_count(
    full_path: str, class_name: str, function_name: str, new_body: str
):
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        source = f.read()

    lines = source.splitlines()

    _, start_line, end_line = read_function_by_name_from_file(
        full_path, class_name, function_name
    )

    # 1) Find the opening brace line
    open_brace_line = start_line
    while "{" not in lines[open_brace_line]:
        open_brace_line += 1

    # 2) Determine indentation from the brace line
    brace_line = lines[open_brace_line]
    prefix = brace_line[: len(brace_line) - len(brace_line.lstrip())]
    body_indent = prefix + "    "

    # 3) Re-indent the new body
    body_lines = []
    for l in new_body.splitlines():
        if l.strip() == "":
            body_lines.append(body_indent.rstrip())
        else:
            body_lines.append(body_indent + l.lstrip())

    # 4) Replace ONLY the body (between braces)
    lines[open_brace_line + 1 : end_line] = body_lines

    # 5) Write back
    new_source = "\n".join(lines) + ("\n" if source.endswith("\n") else "")
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(new_source)
    with open("see.java", "w", encoding="utf-8") as f:
        f.write(new_source)


# Small change: ensure subprocess calls use env variable
def checkout_defects4j(project_id: str, bug_id: str, work_dir: str, env=None):
    version = f"{bug_id}b"
    cmd = [DEFECTS4J_BIN, "checkout", "-p", project_id, "-v", version, "-w", work_dir]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def run_defects4j_test(
    work_dir: str, timeout_seconds: int = 60 * 60, env=None
) -> Tuple[bool, int, str]:
    proc = subprocess.run(
        [DEFECTS4J_BIN, "test"],
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_seconds,
        env=env,
    )
    out = proc.stdout
    failing = 0
    for line in out.splitlines():
        if line.startswith("Failing tests:"):
            try:
                failing = int(line.split(":")[1].strip())
            except Exception:
                pass
    passed = failing == 0
    return passed, failing, out


# Worker function: must be top-level (picklable)
def process_single_task(
    item: dict,
    project_bug_map: Dict[str, str],
    timeout_seconds: int = 60 * 60,
    env_for_subprocess=None,
) -> dict:
    """
    Process a single prediction item: checkout project, patch java file, run tests.
    Returns a dict 'rec' with task_id, passed, failing_tests, log or error.
    """
    rec = {}
    try:
        meta = item.get("metadata", {})
        task_id = meta.get("task_id")
        rec["task_id"] = task_id
        rec["row_id"] = item.get("row_id")
        fpath_tuple = meta.get("fpath_tuple")
        class_name = meta.get("class_name")
        function_name = meta.get("function_name")
        pred_body = item.get("clean_prediction_code", "")

        if not (task_id and fpath_tuple and class_name and function_name):
            rec.update({"passed": False, "error": "Missing metadata fields"})
            return rec

        project_prefix = task_id.split("/")[0]
        if project_prefix not in PROJECT_MAP:
            rec.update(
                {
                    "passed": False,
                    "error": f"Project mapping missing for '{project_prefix}'. Update PROJECT_MAP.",
                }
            )
            return rec

        if project_prefix not in project_bug_map:
            rec.update(
                {
                    "passed": False,
                    "error": f"No bug id mapping for project '{project_prefix}'.",
                }
            )
            return rec

        project_id = PROJECT_MAP[project_prefix]
        bug_id = project_bug_map[project_prefix]

        work_dir = tempfile.mkdtemp(prefix="d4j_")
        backup_java = None
        java_full_path = None
        try:
            # checkout (uses env)
            checkout_defects4j(project_id, bug_id, work_dir, env=env_for_subprocess)

            rel = rel_path_from_fpath_tuple(fpath_tuple)
            java_full_path = os.path.join(work_dir, rel)
            if not os.path.exists(java_full_path):
                filename = fpath_tuple[-1]
                found = None
                for root, _, files in os.walk(work_dir):
                    if filename in files:
                        found = os.path.join(root, filename)
                        break
                if found:
                    java_full_path = found
                else:
                    raise FileNotFoundError(
                        f"Java file not found at {java_full_path} and fallback search failed."
                    )

            rec["java_file"] = java_full_path
            backup_java = java_full_path + ".bak"
            shutil.copyfile(java_full_path, backup_java)
            # replace_method_body_using_brace_count(java_full_path, class_name, function_name, pred_body)

            passed, failing, log = run_defects4j_test(
                work_dir, timeout_seconds=timeout_seconds, env=env_for_subprocess
            )
            rec.update({"passed": passed, "failing_tests": failing, "log": log})
        finally:
            # attempt to restore and cleanup
            try:
                if backup_java and os.path.exists(backup_java):
                    shutil.move(backup_java, java_full_path)
            except Exception:
                pass
            try:
                shutil.rmtree(work_dir)
            except Exception:
                pass

        return rec

    except subprocess.CalledProcessError as cpe:
        return {
            "task_id": item.get("metadata", {}).get("task_id"),
            "row_id": item.get("row_id"),
            "passed": False,
            "error": f"subprocess error: {cpe}",
            "stderr": getattr(cpe, "stderr", None),
        }
    except Exception as e:
        # include traceback for easier debugging in worker
        tb = traceback.format_exc()
        return {
            "task_id": item.get("metadata", {}).get("task_id"),
            "row_id": item.get("row_id"),
            "passed": False,
            "error": str(e),
            "traceback": tb,
        }


def process_predictions_parallel(
    prediction_items: List[dict],
    output_file: str,
    project_bug_map: Dict[str, str],
    max_tasks: int = None,
    workers: int = None,
    timeout_seconds: int = 60 * 60,
):
    """
    Parallel processing coordinator (main process):
    - Submits tasks to ProcessPoolExecutor
    - Writes results to output_file as they complete
    - Shows a tqdm progress bar
    """
    preds = prediction_items
    if max_tasks:
        preds = preds[:max_tasks]
    total = len(preds)
    if total == 0:
        print("No predictions to process.")
        return

    # ensure output dir exists
    out_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # We will write results incrementally
    with open(output_file, "w", encoding="utf-8") as outf, ProcessPoolExecutor(
        max_workers=workers
    ) as exe:
        futures = []
        for item in preds:
            # submit each task; pass env so subprocess calls inside worker can pick it up
            futures.append(
                exe.submit(
                    process_single_task, item, project_bug_map, timeout_seconds, env
                )
            )

        passed_count = 0
        with tqdm(total=total, desc="Processing tasks", unit="task") as pbar:
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception as e:
                    # if a worker crashed in an unhandled way
                    res = {
                        "task_id": None,
                        "passed": False,
                        "error": f"Worker exception: {e}",
                        "traceback": traceback.format_exc(),
                    }

                # write result JSON line
                outf.write(json.dumps(res) + "\n")
                outf.flush()

                if res.get("passed"):
                    passed_count += 1
                pbar.update(1)
                pbar.set_postfix(
                    {"last_task": res.get("task_id"), "passed": passed_count}
                )

    print("====================================")
    print(f"Total tasks processed: {total}")
    print(f"Passed (all tests): {passed_count}")
    print(f"Pass rate: {passed_count/total:.3f}" if total else "N/A")
    print("Results saved to:", output_file)


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Predictions JSONL")
    parser.add_argument("--out", required=True, help="Output JSONL")
    parser.add_argument(
        "--max", type=int, default=None, help="Max tasks for quick testing"
    )
    parser.add_argument(
        "--bug-map",
        type=str,
        default=None,
        help="Optional file mapping project_prefix -> defect4j bug id (JSON or list)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to cpu_count())",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60,
        help="Per-task timeout for defects4j test (seconds)",
    )
    args = parser.parse_args()

    # load predictions
    preds = []
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))

    unique_preds = []
    seen_task_ids = set()

    for item in preds:
        task_id = item.get("metadata", {}).get("task_id").split("/")[0]
        if task_id not in seen_task_ids:
            seen_task_ids.add(task_id)
            unique_preds.append(item)

    preds = unique_preds
    # project_prefix =
    # build project->bug mapping
    project_bug_map = build_project_bug_map(args.bug_map)
    if not project_bug_map:
        print(
            "Warning: project_bug_map is empty. Populate PROJECT_BUG_LIST or pass --bug-map file."
        )

    # run parallel processing
    process_predictions_parallel(
        preds,
        args.out,
        project_bug_map,
        max_tasks=args.max,
        workers=args.workers,
        timeout_seconds=args.timeout,
    )


if __name__ == "__main__":
    main()
