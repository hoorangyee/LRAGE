from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".lrage" / "webapp"


@dataclass
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    output_root: Path = None  # type: ignore[assignment]
    db_path: Path = None  # type: ignore[assignment]
    static_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "static"
    )

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        if self.output_root is None:
            self.output_root = self.data_dir / "eval_results"
        self.output_root = Path(self.output_root).expanduser()
        if self.db_path is None:
            self.db_path = self.data_dir / "webapp.db"
        self.db_path = Path(self.db_path).expanduser()
        self.static_dir = Path(self.static_dir)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
