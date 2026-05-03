import argparse
import csv
from pathlib import Path

IMAGE_COLUMN = "image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join two CSV files.")
    parser.add_argument("first_csv", help="First input CSV file.")
    parser.add_argument("second_csv", help="Second input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        if IMAGE_COLUMN not in reader.fieldnames:
            raise ValueError(f"{path} does not have an '{IMAGE_COLUMN}' column")
        return reader.fieldnames, list(reader)


def join_csv_files(first_csv: Path, second_csv: Path, output: Path) -> None:
    first_header, first_rows = read_csv(first_csv)
    second_header, second_rows = read_csv(second_csv)

    rows: list[dict[str, str]] = []
    seen_images: set[str] = set()
    for row in first_rows + second_rows:
        image = row[IMAGE_COLUMN]
        if image in seen_images:
            continue
        seen_images.add(image)
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=first_header)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    try:
        join_csv_files(
            Path(args.first_csv),
            Path(args.second_csv),
            Path(args.output),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
