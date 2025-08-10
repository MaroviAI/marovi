import importlib
from pathlib import Path


def _reload_config(monkeypatch, data_path=None, use_db=None, db_url=None):
    env = {
        "DATA_STORAGE_PATH": data_path,
        "USE_DATABASE": use_db,
        "DATABASE_URL": db_url,
    }
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, str(value))
    import marovi.config as cfg
    return importlib.reload(cfg)


def test_config_defaults(monkeypatch):
    cfg = _reload_config(monkeypatch)
    expected = Path(cfg.__file__).resolve().parent.parent / "data"
    assert cfg.BASE_PATH == expected
    assert cfg.USE_DATABASE is False
    assert cfg.DATABASE_URL == "sqlite:///database/papers.db"


def test_config_environment_overrides(monkeypatch, tmp_path):
    cfg = _reload_config(
        monkeypatch,
        data_path=tmp_path,
        use_db="true",
        db_url="postgresql://user/db",
    )
    assert cfg.BASE_PATH == tmp_path.resolve()
    assert cfg.USE_DATABASE is True
    assert cfg.DATABASE_URL == "postgresql://user/db"
