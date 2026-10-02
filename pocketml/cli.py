import argparse
import sys

from pocketml import core


def main(argv=None):
    p = argparse.ArgumentParser(prog="pocketml", description="Validated models from your data, on your machine.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="check the data, compare model tiers, save the winner")
    t.add_argument("data")
    t.add_argument("--target", help="column to predict")
    t.add_argument("--keep", action="append", default=[], help="column to keep even if a check flags it")
    t.add_argument("--out", default="runs")
    pr = sub.add_parser("predict", help="predict with a saved run")
    pr.add_argument("run_dir")
    pr.add_argument("data")
    pr.add_argument("--out", default="predictions.csv")
    a = p.parse_args(argv)

    try:
        df = core.load(a.data)
        if a.cmd == "predict":
            core.predict(a.run_dir, df).to_csv(a.out, index=False)
            print(f"wrote {len(df)} predictions to {a.out}")
            return 0
        if not a.target:
            p.error(f"--target is required. Columns: {', '.join(map(str, df.columns))}")
        report, bundle = core.evaluate(df, a.target, keep=a.keep)
        run = core.save(a.out, report, bundle)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(core.to_markdown(report))
    print(f"saved to {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
