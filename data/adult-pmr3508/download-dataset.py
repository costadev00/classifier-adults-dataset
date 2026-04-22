import os
from pathlib import Path
from zipfile import ZipFile

import requests


def load_kaggle_credentials(env_path: Path = Path(".env")) -> None:
    if os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    ):
        return

    if not env_path.exists():
        return

    env_values = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_values[key.strip()] = value.strip().strip("\"'")

    username = env_values.get("KAGGLE_USERNAME") or env_values.get("kaggle.username")
    token = env_values.get("KAGGLE_API_TOKEN") or env_values.get("kaggle.api_token")
    secret = env_values.get("KAGGLE_KEY") or env_values.get("kaggle.key")

    if username:
        os.environ.setdefault("KAGGLE_USERNAME", username)
    if token:
        os.environ.setdefault("KAGGLE_API_TOKEN", token)
    elif secret:
        env_name = "KAGGLE_API_TOKEN" if secret.startswith("KGAT_") else "KAGGLE_KEY"
        os.environ.setdefault(env_name, secret)


def extract_competition_archive(archive_path: Path, destination: Path) -> None:
    with ZipFile(archive_path) as archive:
        archive.extractall(destination)


competition = "adult-pmr3508"
download_dir = Path("data") / competition
archive_path = download_dir / f"{competition}.zip"
download_dir.mkdir(parents=True, exist_ok=True)

load_kaggle_credentials()
has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
has_legacy_key = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
if not has_token and not has_legacy_key:
    raise RuntimeError(
        "Missing Kaggle credentials. Set KAGGLE_API_TOKEN or KAGGLE_USERNAME and KAGGLE_KEY, "
        "or add kaggle.api_token / kaggle.key credentials to .env."
    )

# from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# api.authenticate()
# try:
#     api.competition_download_files(competition, path=str(download_dir), quiet=False)
# except requests.HTTPError as exc:
#     if exc.response is not None and exc.response.status_code == 401:
#         raise RuntimeError(
#             "Kaggle authentication reached the API, but access was denied for this competition. "
#             "Check that the .env token or key is valid and that this Kaggle account has accepted "
#             f"the competition rules for '{competition}'."
#         ) from exc
#     raise

extract_competition_archive(archive_path, download_dir)

print("Path to competition files:", download_dir.resolve())
