import os
import re
import sys
import uuid
import argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_sort_key(filename: str) -> tuple:
    """Extract numeric sort key from filename for stable ordering.

    Priority: files already named with folder prefix sorted by number first,
    then remaining files sorted by any embedded number, then alphabetically.
    """
    nums = re.findall(r'(\d+)', filename)
    if nums:
        return (0, int(nums[0]), filename.lower())
    return (1, 0, filename.lower())


def reindex_folder(folder_path, dry_run=False, target_ext=".jpeg"):
    folder_name = os.path.basename(os.path.normpath(folder_path))

    entries = []
    for e in os.listdir(folder_path):
        full = os.path.join(folder_path, e)
        if not os.path.isfile(full):
            continue
        ext = os.path.splitext(e)[1].lower()
        if ext in IMG_EXTS:
            entries.append(e)

    if not entries:
        print(f"  [SKIP] No image files in: {folder_path}")
        return

    prefix_pat = re.compile(r'^' + re.escape(folder_name) + r'(\d+)', re.IGNORECASE)
    prefixed = []
    others = []
    for e in entries:
        m = prefix_pat.match(e)
        if m:
            prefixed.append((int(m.group(1)), e))
        else:
            others.append(e)

    prefixed.sort(key=lambda x: x[0])
    others.sort(key=lambda x: extract_sort_key(x))

    sorted_entries = [name for _, name in prefixed] + [name for name in others]

    print(f"\n  Folder: {folder_path}  ({len(sorted_entries)} images)")
    if others:
        print(f"  Non-standard names (will be included): {len(others)} files")

    renames = []
    for i, old_name in enumerate(sorted_entries, start=1):
        new_name = f"{folder_name}{i:03d}{target_ext}"
        renames.append((old_name, new_name))

    for old, new in renames:
        marker = " *" if old != new else ""
        print(f"    {old:<55} ->  {new}{marker}")

    if dry_run:
        return

    final_names = [new for _, new in renames]
    if len(final_names) != len(set(final_names)):
        print("  ERROR: Duplicate target names detected. Aborting this folder.")
        return

    temp_map = []
    for old, _ in renames:
        src = os.path.join(folder_path, old)
        tmp = os.path.join(folder_path, f".__tmp__{uuid.uuid4().hex}__")
        os.rename(src, tmp)
        temp_map.append((tmp, old))

    old_to_new = dict(renames)
    for tmp, old in temp_map:
        new = old_to_new[old]
        dst = os.path.join(folder_path, new)
        os.rename(tmp, dst)

    print(f"  Done. {len(renames)} files renamed to {folder_name}001-{folder_name}{len(renames):03d}{target_ext}")


def main():
    parser = argparse.ArgumentParser(
        description="Reindex ALL image files inside subfolders to folder001.jpeg, folder002.jpeg, ..."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root folder containing subfolders (default: current directory)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview renames without applying changes"
    )
    parser.add_argument(
        "--ext",
        default=".jpeg",
        help="Target extension for all renamed files (default: .jpeg)"
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        help="Specific subfolders to process (default: all subfolders under root)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Error: '{args.root}' is not a valid directory.")
        sys.exit(1)

    if args.folders:
        targets = [os.path.join(args.root, f) for f in args.folders]
    else:
        targets = [
            os.path.join(args.root, d)
            for d in sorted(os.listdir(args.root))
            if os.path.isdir(os.path.join(args.root, d))
        ]

    if not targets:
        print("No subfolders found.")
        sys.exit(0)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(targets)} folder(s) under: {args.root}")

    for folder in targets:
        if not os.path.isdir(folder):
            print(f"  [SKIP] Not a folder: {folder}")
            continue
        reindex_folder(folder, dry_run=args.dry_run, target_ext=args.ext)

    if args.dry_run:
        print("\nDry run complete. No files were renamed.")
        print("Run without --dry-run to apply changes.")
    else:
        print("\nAll done.")


if __name__ == "__main__":
    main()
