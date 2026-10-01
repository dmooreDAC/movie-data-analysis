import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from movies import DATA_PATH, explode_genres

CHART_PATH = Path(__file__).resolve().parent / "png" / "genre_satisfaction.png"
JSON_PATH = Path(__file__).resolve().parent / "json" / "genre_satisfaction.json"
SMALL_N_THRESHOLD = 30
RATING_SCALE = [1, 5]


def movie_level_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("movie_id", as_index=False)
        .agg(
            avg_rating=("rating", "mean"),
            n_ratings=("rating", "size"),
            genres=("genres", "first"),
        )
    )


def genre_satisfaction(
    ratings: pd.DataFrame, small_n_threshold: int = SMALL_N_THRESHOLD
) -> tuple[pd.DataFrame, float]:
    movies = movie_level_stats(ratings)
    overall_movie_mean = float(movies["avg_rating"].mean())
    exploded = explode_genres(movies)

    by_genre = (
        exploded.groupby("genre", as_index=False)
        .agg(
            avg_rating=("avg_rating", "mean"),
            n_movies=("movie_id", "nunique"),
            n_ratings=("n_ratings", "sum"),
        )
        .sort_values("avg_rating", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    by_genre["delta_vs_overall"] = by_genre["avg_rating"] - overall_movie_mean
    by_genre["small_n"] = by_genre["n_movies"] < small_n_threshold
    return by_genre, overall_movie_mean


def satisfaction_payload(
    by_genre: pd.DataFrame, overall_movie_mean: float, small_n_threshold: int = SMALL_N_THRESHOLD
) -> dict:
    genres = []
    for row in by_genre.itertuples(index=False):
        genres.append(
            {
                "genre": row.genre,
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
        "method": "mean of per-movie means; hybrids exploded",
        "small_n_threshold": small_n_threshold,
        "genres": genres,
    }


def plot_genre_satisfaction(by_genre: pd.DataFrame, overall_movie_mean: float) -> None:
    plot_df = by_genre.sort_values("avg_rating", ascending=True, kind="mergesort")
    colors = ["#8aa8c4" if small else "#3b6ea5" for small in plot_df["small_n"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["genre"], plot_df["avg_rating"], color=colors)
    ax.axvline(overall_movie_mean, color="#c45c26", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Average movie rating (1–5)")
    ax.set_ylabel("Genre")
    ax.set_title("Genre satisfaction: mean of per-movie averages")
    ax.set_xlim(1, 5.35)
    for y, avg, n, small in zip(
        range(len(plot_df)), plot_df["avg_rating"], plot_df["n_movies"], plot_df["small_n"]
    ):
        note = "  | small n" if small else ""
        ax.text(avg + 0.04, y, f"{avg:.1f}  ({n} titles{note})", va="center", fontsize=8)
    ax.annotate(
        f"Orange line: catalog mean of movie averages ({overall_movie_mean:.2f})  |  "
        f"lighter bars: fewer than {SMALL_N_THRESHOLD} titles",
        xy=(0, -0.12),
        xycoords="axes fraction",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


def main() -> None:
    ratings = pd.read_csv(DATA_PATH)
    by_genre, overall_movie_mean = genre_satisfaction(ratings)
    payload = satisfaction_payload(by_genre, overall_movie_mean)

    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Catalog mean (each movie equal): {overall_movie_mean:.2f} / 5")
    print("Method: mean of per-movie means; hybrids count toward each genre.")
    print()
    print(
        by_genre.to_string(
            index=False,
            formatters={
                "avg_rating": "{:.2f}".format,
                "delta_vs_overall": "{:+.2f}".format,
            },
        )
    )
    highest = by_genre.iloc[0]
    lowest = by_genre.iloc[-1]
    print()
    print(
        f"Highest: {highest.genre} ({highest.avg_rating:.1f})  |  "
        f"Lowest: {lowest.genre} ({lowest.avg_rating:.1f})"
    )
    print(f"JSON saved to {JSON_PATH}")

    plot_genre_satisfaction(by_genre, overall_movie_mean)
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
