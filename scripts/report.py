"""Generate HTML visualization reports for recon/runner logs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "logs"


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for exploration/execution logs")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Recon mode
    p_recon = sub.add_parser("recon", help="Page exploration report")
    p_recon.add_argument("--app", required=True, help="App name under logs/recon/")
    p_recon.add_argument("--page", default=None, help="Specific page name (default: all pages)")

    # Runner mode
    p_runner = sub.add_parser("runner", help="Policy runner report")
    p_runner.add_argument("--run", required=True, type=Path, help="Run directory under logs/policy_expr/")

    args = parser.parse_args()

    from scripts.report_builder import (
        ReconReportBuilder,
        RunnerReportBuilder,
        save_recon_report,
        save_report,
    )

    if args.mode == "recon":
        log_dir = LOGS / "recon" / args.app
        if args.page:
            log_dir = log_dir / args.page
        builder = ReconReportBuilder()
        data = builder.build(log_dir)
        output = (LOGS / "recon" / args.app / "report.html")

        path = save_recon_report(data, output)

    elif args.mode == "runner":
        run_dir = args.run.resolve()
        if not run_dir.is_absolute():
            run_dir = (LOGS / "policy_expr" / run_dir).resolve()
        builder = RunnerReportBuilder()
        data = builder.build(run_dir)
        output = run_dir / "report.html"

        path = save_report(data, output, grid=True)
    else:
        raise AssertionError(f"unknown mode: {args.mode}")

    print(f"Report saved: {path}")


if __name__ == "__main__":
    main()
