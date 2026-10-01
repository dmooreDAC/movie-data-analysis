from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "movie_ratings.csv"
CHART_PATH = Path(__file__).resolve().parent / "genre_distribution.png"
INVALID_GENRES = frozenset({"", "(no genres listed)", "unknown"})


def explode_genres(df: pd.DataFrame, genres_col: str = "genres") -> pd.DataFrame:
    """One row per genre tag; drop empty, unknown, and placeholder labels."""
    exploded = df.assign(
        genre=df[genres_col].fillna("").astype(str).str.split("|")
    ).explode("genre", ignore_index=True)
    exploded["genre"] = exploded["genre"].str.strip()
    return exploded[~exploded["genre"].isin(INVALID_GENRES)].copy()


def rated_movie_genres(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Unique rated titles, exploded genre counts, and share of multi-genre films."""
    movies = ratings.drop_duplicates(subset="movie_id")[["movie_id", "title", "genres"]].copy()
    movies["genres"] = movies["genres"].fillna("").astype(str).str.strip()

    n_titles = len(movies)
    multi_genre_share = movies["genres"].str.contains(r"\|", regex=True).mean()

    exploded = explode_genres(movies)

    counts = exploded["genre"].value_counts().rename_axis("genre").reset_index(name="n_movies")
    counts["pct_of_rated_titles"] = 100 * counts["n_movies"] / n_titles
    return movies, counts, float(multi_genre_share)


def plot_genre_distribution(counts: pd.DataFrame, n_titles: int, multi_genre_share: float) -> None:
    plot_df = counts.sort_values("n_movies", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["genre"], plot_df["pct_of_rated_titles"], color="#3b6ea5")
    ax.set_xlabel("% of rated movies tagged with this genre")
    ax.set_ylabel("Genre")
    ax.set_title("Genre mix among uniquely rated movies")
    ax.set_xlim(0, plot_df["pct_of_rated_titles"].max() * 1.15)
    for y, pct, n in zip(
        range(len(plot_df)), plot_df["pct_of_rated_titles"], plot_df["n_movies"]
    ):
        ax.text(pct + 0.4, y, f"{pct:.1f}%  ({n})", va="center", fontsize=8)
    ax.annotate(
        f"{n_titles:,} unique rated titles  ·  "
        f"{multi_genre_share:.0%} have 2+ genres  ·  percents sum > 100%",
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
    movies, counts, multi_genre_share = rated_movie_genres(ratings)
    n_titles = len(movies)

    print(f"Unique rated movies: {n_titles:,}")
    print(f"Share with multiple genres: {multi_genre_share:.1%}")
    print()
    print(counts.to_string(index=False, formatters={"pct_of_rated_titles": "{:.1f}%".format}))
    print()
    print("Note: hybrids count toward each tag, so percents sum to more than 100%.")

    plot_genre_distribution(counts, n_titles, multi_genre_share)
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
