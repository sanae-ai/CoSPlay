import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


DEFAULT_REPO_ID = "yomi017/CosPlay"
DATA_PREFIX = "Datasets/CURE_data"

# Remote dataset layout:
#   Datasets/CURE_data/Full_Dataset/chunked/     full-dataset benchmark shards
#   Datasets/CURE_data/Full_Dataset/complete/    four complete full-dataset files
#   Datasets/CURE_data/Small_Dataset/            small-dataset shards
GROUP_PREFIXES = {
    "full-dataset-chunked": f"{DATA_PREFIX}/Full_Dataset/chunked",
    "full-dataset-complete": f"{DATA_PREFIX}/Full_Dataset/complete",
    "full-dataset": f"{DATA_PREFIX}/Full_Dataset",
    "small-dataset": f"{DATA_PREFIX}/Small_Dataset",
}
GROUP_ALIASES = {
    # Backward-compatible names used by older README examples and scripts.
    "main-chunked": "full-dataset-chunked",
    "main-full": "full-dataset-complete",
    "main": "full-dataset",
    "generalization": "small-dataset",
    "shards": "full-dataset-chunked",
    "full": "full-dataset-complete",
    "small": "small-dataset",
}

# These four files are the complete Full Dataset benchmark dumps. They are useful
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
        return normalize_legacy_remote_path(name)
    if name.startswith("Datasets/"):
        return normalize_legacy_remote_path(name)
    if name.startswith(("Full_Dataset/", "Small_Dataset/")):
        return f"{DATA_PREFIX}/{name}"
    if name.startswith(("main/", "generalization/")):
        return normalize_legacy_remote_path(f"{DATA_PREFIX}/{name}")

    if not name.endswith(".json"):
        name = f"{name}.json"

    if name.startswith("LB_LCB_CC_CF_200"):
        return f"{GROUP_PREFIXES['small-dataset']}/{name}"
    if name in FULL_DATASETS:
        return f"{GROUP_PREFIXES['full-dataset-complete']}/{name}"
    return f"{GROUP_PREFIXES['full-dataset-chunked']}/{name}"


def normalize_legacy_remote_path(path: str) -> str:
    """Map older public CURE_data paths to the Full/Small Dataset layout."""
    legacy_prefixes = {
        f"{DATA_PREFIX}/main/chunked/": f"{GROUP_PREFIXES['full-dataset-chunked']}/",
        f"{DATA_PREFIX}/main/full/": f"{GROUP_PREFIXES['full-dataset-complete']}/",
        f"{DATA_PREFIX}/main/": f"{GROUP_PREFIXES['full-dataset']}/",
        f"{DATA_PREFIX}/generalization/": f"{GROUP_PREFIXES['small-dataset']}/",
    }
    for old_prefix, new_prefix in legacy_prefixes.items():
        if path.startswith(old_prefix):
            return f"{new_prefix}{path.removeprefix(old_prefix)}"
    return path


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
    """Split full-dataset chunked/complete files from small-dataset files."""
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
        choices=[
            "full-dataset-chunked",
            "full-dataset-complete",
            "full-dataset",
            "small-dataset",
            "all",
            "main-chunked",
            "main-full",
            "main",
            "generalization",
            "shards",
            "full",
            "small",
        ],
        default="full-dataset-chunked",
        help=(
            "Which dataset group to download when --dataset is omitted. "
            "'full-dataset-chunked' downloads Full Dataset shards; "
            "'full-dataset-complete' downloads the four complete Full Dataset "
            "files; 'small-dataset' downloads Small Dataset shards; "
            "'full-dataset' downloads both Full Dataset groups; 'all' "
            "downloads every CURE_data file. Legacy aliases: 'main-chunked' "
            "and 'shards' = 'full-dataset-chunked', 'main-full' and 'full' "
            "= 'full-dataset-complete', 'generalization' and 'small' = "
            "'small-dataset'. Default: full-dataset-chunked."
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
