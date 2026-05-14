import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "yomi017/CosPlay"
DATA_PREFIX = "Datasets/CURE_data"

# Remote dataset layout:
#   Datasets/CURE_data/main/chunked/      main-table benchmark shards
#   Datasets/CURE_data/main/full/         four complete main-table benchmarks
#   Datasets/CURE_data/generalization/    small-table/generalization shards
GROUP_PREFIXES = {
    "main-chunked": f"{DATA_PREFIX}/main/chunked",
    "main-full": f"{DATA_PREFIX}/main/full",
    "main": f"{DATA_PREFIX}/main",
    "generalization": f"{DATA_PREFIX}/generalization",
}
GROUP_ALIASES = {
    # Backward-compatible names used by older README examples.
    "shards": "main-chunked",
    "full": "main-full",
}

# These four files are the complete main-table benchmark dumps. They are useful
# for full reprocessing, but they are much larger than the chunked files used by
# the default evaluation scripts, so they are not downloaded by default.
FULL_DATASETS = {
    "CodeContests.json",
    "CodeForces.json",
    "LiveBench.json",
    "LiveCodeBench.json",
}


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "CURE_data"


def normalize_dataset_path(name: str) -> str:
    name = name.replace("\\", "/")
    if name.startswith(f"{DATA_PREFIX}/"):
        return name
    if name.startswith("Datasets/"):
        return name
    if name.startswith(("main/", "generalization/")):
        return f"{DATA_PREFIX}/{name}"

    if not name.endswith(".json"):
        name = f"{name}.json"

    if name.startswith("LB_LCB_CC_CF_200"):
        return f"{GROUP_PREFIXES['generalization']}/{name}"
    if name in FULL_DATASETS:
        return f"{GROUP_PREFIXES['main-full']}/{name}"
    return f"{GROUP_PREFIXES['main-chunked']}/{name}"


def list_dataset_files(repo_id: str, revision: str | None) -> list[str]:
    """Return all JSON files under the public Datasets/CURE_data folder."""
    api = HfApi()
    entries = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=DATA_PREFIX,
        recursive=True,
        revision=revision,
    )
    return sorted(entry.path for entry in entries if entry.path.endswith(".json"))


def filter_by_group(paths: list[str], group: str) -> list[str]:
    """Split main-table chunked/full files from generalization files."""
    group = GROUP_ALIASES.get(group, group)

    if group == "all":
        return paths
    if group in GROUP_PREFIXES:
        return [path for path in paths if path.startswith(f"{GROUP_PREFIXES[group]}/")]

    raise ValueError(f"Unknown dataset group: {group}")


def download_one(
    repo_id: str,
    path_in_repo: str,
    output_dir: Path,
    revision: str | None,
    cache_dir: Path | None,
    force: bool,
) -> Path:
    # hf_hub_download first resolves the file into the Hugging Face cache.
    # We then copy the cached file into CURE_data/, which is where evaluation
    # scripts in this repo expect to find <dataset>.json.
    cached_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=path_in_repo,
            revision=revision,
            cache_dir=cache_dir,
        )
    )
    output_path = output_dir / Path(path_in_repo).name
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not force and output_path.stat().st_size == cached_path.stat().st_size:
        print(f"Skip existing: {output_path}")
        return output_path

    shutil.copy2(cached_path, output_path)
    print(f"Downloaded: {path_in_repo} -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CoSPlay benchmark datasets from Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo id. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help=(
            "Dataset file stem(s) or .json file name(s) to download, e.g. "
            "LiveBench_chunk_0 CodeForces. Overrides --group when provided."
        ),
    )
    parser.add_argument(
        "--group",
        choices=["main-chunked", "main-full", "main", "generalization", "all", "shards", "full"],
        default="main-chunked",
        help=(
            "Which dataset group to download when --dataset is omitted. "
            "'main-chunked' downloads main-table shards; 'main-full' downloads "
            "the four complete main-table files; 'generalization' downloads "
            "small-table/generalization shards; 'main' downloads both main "
            "groups; 'all' downloads every CURE_data file. Legacy aliases: "
            "'shards' = 'main-chunked', 'full' = 'main-full'. "
            "Default: main-chunked."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Directory to write JSON files. Default: ../CURE_data from this script.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit SHA.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset files and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local files even when the size already matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        paths = list_dataset_files(args.repo_id, args.revision)
        paths = paths if args.dataset else filter_by_group(paths, args.group)
        for path in paths:
            print(path.removeprefix(f"{DATA_PREFIX}/"))
        return

    if args.dataset:
        paths = [normalize_dataset_path(name) for name in args.dataset]
    else:
        paths = filter_by_group(
            list_dataset_files(args.repo_id, args.revision),
            args.group,
        )

    if not paths:
        raise RuntimeError("No dataset files found to download.")

    print(f"Repo: {args.repo_id}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Group: {'custom' if args.dataset else args.group}")
    print(f"Files: {len(paths)}")

    for index, path_in_repo in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path_in_repo}")
        download_one(
            repo_id=args.repo_id,
            path_in_repo=path_in_repo,
            output_dir=args.output_dir,
            revision=args.revision,
            cache_dir=args.cache_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()
