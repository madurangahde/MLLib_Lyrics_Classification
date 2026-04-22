import argparse
import csv
from pathlib import Path


def is_valid_year(value: str, min_year: int, max_year: int) -> bool:
    if value is None:
        return False

    year_str = str(value).strip()
    if not year_str:
        return False

    if not year_str.isdigit():
        return False

    year = int(year_str)
    return min_year <= year <= max_year


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether every record in a CSV has a valid year value."
    )
    parser.add_argument(
        "--csv",
        default="data/Student_dataset.csv",
        help="Path to the CSV file (default: data/Student_dataset.csv)",
    )
    parser.add_argument(
        "--year-column",
        default="release_date",
        help="Column name that should contain the year (default: release_date)",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=1000,
        help="Minimum accepted year (default: 1000)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2100,
        help="Maximum accepted year (default: 2100)",
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

        if args.year_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{args.year_column}' not found. Available columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            total += 1
            year_value = row.get(args.year_column)
            artist = (row.get("artist_name") or "").strip()
            track = (row.get("track_name") or "").strip()

            if is_valid_year(year_value, args.min_year, args.max_year):
                ok_count += 1
                print(
                    f"Row {row_num}: OK | year={year_value} | artist='{artist}' | track='{track}'"
                )
            else:
                not_ok_count += 1
                print(
                    f"Row {row_num}: NOT OK | year={year_value} | artist='{artist}' | track='{track}'"
                )

    print("\n--- SUMMARY ---")
    print(f"CSV file       : {csv_path}")
    print(f"Year column    : {args.year_column}")
    print(f"Accepted range : {args.min_year}-{args.max_year}")
    print(f"Total rows     : {total}")
    print(f"OK rows        : {ok_count}")
    print(f"NOT OK rows    : {not_ok_count}")
    print(f"All rows valid : {'YES' if not_ok_count == 0 else 'NO'}")


if __name__ == "__main__":
    main()
