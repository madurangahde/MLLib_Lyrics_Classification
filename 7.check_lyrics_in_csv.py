import argparse
import csv
from pathlib import Path


def is_valid_lyrics(value: str, min_chars: int, require_alpha: bool) -> bool:
    if value is None:
        return False

    text = str(value).strip()
    if not text:
        return False

    if len(text) < min_chars:
        return False

    if require_alpha and not any(ch.isalpha() for ch in text):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether every record in a CSV has valid lyrics text."
    )
    parser.add_argument(
        "--csv",
        default="data/Student_dataset.csv",
        help="Path to the CSV file (default: data/Student_dataset.csv)",
    )
    parser.add_argument(
        "--lyrics-column",
        default="lyrics",
        help="Column name that should contain lyrics (default: lyrics)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Minimum number of non-space characters required (default: 1)",
    )
    parser.add_argument(
        "--require-alpha",
        action="store_true",
        help="Require at least one alphabetic character in lyrics",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    total = 0
    ok_count = 0
    not_ok_count = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV has no header row.")

        if args.lyrics_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{args.lyrics_column}' not found. Available columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            total += 1
            lyrics_value = row.get(args.lyrics_column)
            artist = (row.get("artist_name") or "").strip()
            track = (row.get("track_name") or "").strip()

            valid = is_valid_lyrics(
                lyrics_value,
                min_chars=args.min_chars,
                require_alpha=args.require_alpha,
            )

            lyrics_len = len((lyrics_value or "").strip())

            if valid:
                ok_count += 1
                print(
                    f"Row {row_num}: OK | lyrics_len={lyrics_len} | artist='{artist}' | track='{track}'"
                )
            else:
                not_ok_count += 1
                print(
                    f"Row {row_num}: NOT OK | lyrics_len={lyrics_len} | artist='{artist}' | track='{track}'"
                )

    print("\n--- SUMMARY ---")
    print(f"CSV file        : {csv_path}")
    print(f"Lyrics column   : {args.lyrics_column}")
    print(f"Min chars       : {args.min_chars}")
    print(f"Require alpha   : {'YES' if args.require_alpha else 'NO'}")
    print(f"Total rows      : {total}")
    print(f"OK rows         : {ok_count}")
    print(f"NOT OK rows     : {not_ok_count}")
    print(f"All rows valid  : {'YES' if not_ok_count == 0 else 'NO'}")


if __name__ == "__main__":
    main()
