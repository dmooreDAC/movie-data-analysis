import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from movies import DATA_PATH

CHART_PATH = Path(__file__).resolve().parent / "ratings_over_time.png"
JSON_PATH = Path(__file__).resolve().parent / "ratings_over_time.json"
SMALL_N_THRESHOLD = 30
RATING_SCALE = [1, 5]
CAVEATS = [
    "Votes were collected around 1997-1998; this is not rating-behavior over calendar time.",
    "Older decades are a survivor catalog; recent decades include more average titles.",
]


def movies_with_decade(ratings: pd.DataFrame) -> pd.DataFrame:
    movies = ratings.groupby("movie_id", as_index=False).agg(
        avg_rating=("rating", "mean"),
        n_ratings=("rating", "size"),
        year=("year", "first"),
        decade=("decade", "first"),
    )
    movies = movies.dropna(subset=["year", "decade"])
    movies["decade"] = movies["decade"].astype(int)
    movies["year"] = movies["year"].astype(float)
    return movies[(movies["decade"] > 0) & (movies["year"] > 1800)].copy()


def ratings_by_release_decade(
    ratings: pd.DataFrame, small_n_threshold: int = SMALL_N_THRESHOLD
) -> tuple[pd.DataFrame, float]:
    movies = movies_with_decade(ratings)
    overall_movie_mean = float(movies["avg_rating"].mean())
    by_decade = (
        movies.groupby("decade", as_index=False)
        .agg(
            avg_rating=("avg_rating", "mean"),
            n_movies=("movie_id", "nunique"),
            n_ratings=("n_ratings", "sum"),
        )
        .sort_values("decade", kind="mergesort")
        .reset_index(drop=True)
    )
    by_decade["label"] = by_decade["decade"].map(lambda d: f"{d}s")
    by_decade["delta_vs_overall"] = by_decade["avg_rating"] - overall_movie_mean
    by_decade["small_n"] = by_decade["n_movies"] < small_n_threshold
    return by_decade, overall_movie_mean


def ratings_over_time_payload(
    by_decade: pd.DataFrame,
    overall_movie_mean: float,
    small_n_threshold: int = SMALL_N_THRESHOLD,
) -> dict:
    decades = []
    for row in by_decade.itertuples(index=False):
        decades.append(
            {
                "decade": int(row.decade),
                "label": row.label,
                "avg_rating": round(float(row.avg_rating), 2),
                "n_movies": int(row.n_movies),
                "n_ratings": int(row.n_ratings),
                "delta_vs_overall": round(float(row.delta_vs_overall), 2),
                "small_n": bool(row.small_n),
            }
        )
    return {
        "overall_movie_mean": round(overall_movie_mean, 2),
        "rating_scale": RATING_SCALE,
        "x_axis": "movie_release_decade",
        "method": "mean of per-movie means by release decade",
        "small_n_threshold": small_n_threshold,
        "caveats": CAVEATS,
        "decades": decades,
    }


def plot_ratings_over_time(by_decade: pd.DataFrame, overall_movie_mean: float) -> None:
    colors = ["#8aa8c4" if small else "#3b6ea5" for small in by_decade["small_n"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(by_decade))
    ax.bar(x, by_decade["avg_rating"], color=colors)
    ax.axhline(overall_movie_mean, color="#c45c26", linestyle="--", linewidth=1.5)
    ax.set_xticks(list(x), by_decade["label"])
    ax.set_ylim(1, 5)
    ax.set_xlabel("Movie release decade")
    ax.set_ylabel("Average movie rating (1-5)")
    ax.set_title("Ratings over time: satisfaction by release decade")
    for i, avg, n, small in zip(
        x, by_decade["avg_rating"], by_decade["n_movies"], by_decade["small_n"]
    ):
        note = "*" if small else ""
        ax.text(i, avg + 0.06, f"{avg:.1f}{note}\n({n})", ha="center", va="bottom", fontsize=8)
    ax.annotate(
        f"Orange line: catalog mean ({overall_movie_mean:.2f})  |  "
        f"lighter bars / * : fewer than {SMALL_N_THRESHOLD} titles  |  "
        "older decades are a survivor shelf",
        xy=(0, -0.16),
        xycoords="axes fraction",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    ratings = pd.read_csv(DATA_PATH)
    by_decade, overall_movie_mean = ratings_by_release_decade(ratings)
    payload = ratings_over_time_payload(by_decade, overall_movie_mean)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Release decade, not date of the vote. Each movie counts once.")
    print(f"Catalog mean (valid-year titles): {overall_movie_mean:.2f} / 5")
    for caveat in CAVEATS:
        print(f"- {caveat}")
    print()
    print(
        by_decade[["decade", "label", "avg_rating", "n_movies", "n_ratings", "delta_vs_overall", "small_n"]].to_string(
            index=False,
            formatters={
                "avg_rating": "{:.2f}".format,
                "delta_vs_overall": "{:+.2f}".format,
            },
        )
    )
    peak = by_decade.loc[by_decade["avg_rating"].idxmax()]
    trough = by_decade.loc[by_decade["avg_rating"].idxmin()]
    print()
    print(
        f"Highest era: {peak.label} ({peak.avg_rating:.1f})  |  "
        f"Lowest era: {trough.label} ({trough.avg_rating:.1f})"
    )
    print(f"JSON saved to {JSON_PATH}")
    plot_ratings_over_time(by_decade, overall_movie_mean)
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
