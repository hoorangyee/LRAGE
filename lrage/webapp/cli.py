"""`lrage-serve` entry point.

Run state (job queue, progress buffers) is in-process, so this always runs a
single uvicorn worker process.
"""
import argparse
import webbrowser

from lrage.webapp.settings import DEFAULT_DATA_DIR, Settings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lrage-serve", description="Serve the LRAGE web UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory for the run index database (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for evaluation outputs (default: <data-dir>/eval_results)",
    )
    parser.add_argument(
        "--open-browser", action="store_true", help="Open the UI in a browser"
    )
    args = parser.parse_args()

    try:
        import uvicorn

        from lrage.webapp.app import create_app
    except ImportError as e:
        raise SystemExit(
            f"Missing web dependencies ({e.name}). Install with: pip install lrage[web]"
        )

    settings = Settings(data_dir=args.data_dir, output_root=args.output_dir)
    app = create_app(settings)

    if args.open_browser:
        webbrowser.open(f"http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
