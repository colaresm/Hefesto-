import argparse
from pathlib import Path

from hefesto.samples import make_bga_fatigue_doe, to_workbench_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Hefesto demo DOE dataset (BGA solder fatigue)."
    )
    parser.add_argument("--n", type=int, default=200, help="Number of design points")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("bga_solder_fatigue_doe.csv")
    )
    args = parser.parse_args()
    args.out.write_text(to_workbench_csv(make_bga_fatigue_doe(args.n, args.seed)), encoding="utf-8")
    print(f"Wrote {args.out}")
