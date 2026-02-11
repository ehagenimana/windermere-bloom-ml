import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="windermere-project",
        description="Lake Windermere bloom risk prediction pipeline (MLOps-first).",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version info.",
    )
    args = parser.parse_args()

    if args.version:
        print("windermere-project 0.1.0")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
