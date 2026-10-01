import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from movies import DATA_PATH

CHART_PATH = Path(__file__).resolve().parent / "top_movies.png"
JSON_PATH = Path(__file__).resolve().parent / "top_movies.json"
RATING_SCALE = [1, 5]
FLOORS = [50, 150]
TOP_N = 5
BAYES_PRIOR_VOTES = 50  # "m": how many votes of evidence a movie needs to outweigh the prior


def movie_level_stats(ratings: pd.DataFrame) -> pd.DataFrame:
    return (
        ratings.groupby("movie_id", as_index=False)
        .agg(
            title=("title", "first"),
            avg_rating=("rating", "mean"),
            n_ratings=("rating", "size"),
            pct_five_star=("rating", lambda s: 100 * (s == 5).mean()),
        )
    )


def add_bayesian_score(
    movies: pd.DataFrame, prior_votes: int = BAYES_PRIOR_VOTES, prior_mean: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Weighted rating: (v / (v + m)) * R + (m / (v + m)) * C.

    C is the mean of all individual ratings, so movies with few votes are pulled
    toward the typical rating instead of being cut off by a hard floor.
    """
    if prior_mean is None:
        prior_mean = float(
            (movies["avg_rating"] * movies["n_ratings"]).sum() / movies["n_ratings"].sum()
        )
    v = movies["n_ratings"]
    movies = movies.assign(
        bayes_score=(v / (v + prior_votes)) * movies["avg_rating"]
        + (prior_votes / (v + prior_votes)) * prior_mean
    )
    return movies, prior_mean


def top_with_floor(movies: pd.DataFrame, floor: int, top_n: int = TOP_N) -> pd.DataFrame:
    """Top titles by plain average, counting only movies with at least `floor` ratings."""
    qualified = movies[movies["n_ratings"] >= floor]
    top = (
        qualified.sort_values(["avg_rating", "n_ratings"], ascending=[False, False], kind="mergesort")
        .head(top_n)
        .reset_index(drop=True)
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def top_bayesian(movies: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    top = (
        movies.sort_values(["bayes_score", "n_ratings"], ascending=[False, False], kind="mergesort")
        .head(top_n)
        .reset_index(drop=True)
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    return top


def rows_payload(df: pd.DataFrame) -> list[dict]:
    rows = []
    for row in df.itertuples(index=False):
        entry = {
            "rank": int(row.rank),
            "title": row.title,
            "avg_rating": round(float(row.avg_rating), 2),
            "n_ratings": int(row.n_ratings),
            "pct_five_star": round(float(row.pct_five_star), 1),
        }
        if hasattr(row, "bayes_score"):
            entry["bayes_score"] = round(float(row.bayes_score), 3)
        rows.append(entry)
    return rows


def top_movies_payload(
    floor_tables: dict[int, pd.DataFrame],
    qualified_counts: dict[int, int],
    bayes_table: pd.DataFrame,
    prior_mean: float,
) -> dict:
    return {
        "rating_scale": RATING_SCALE,
        "top_n": TOP_N,
        "method": "per-movie mean rating; ties broken by rating count",
        "floors": [
            {
                "min_ratings": floor,
                "n_qualifying_movies": qualified_counts[floor],
                "top": rows_payload(table),
            }
            for floor, table in floor_tables.items()
        ],
        "bayesian": {
            "prior_votes": BAYES_PRIOR_VOTES,
            "prior_mean": round(prior_mean, 3),
            "formula": "(v/(v+m))*R + (m/(v+m))*C",
            "top": rows_payload(bayes_table),
        },
        "caveats": [
            "Votes were collected around 1997-1998; scores reflect that audience.",
            "Older titles are a survivor catalog; people mostly rate classics they expect to like.",
        ],
    }


def plot_top_movies(
    floor_tables: dict[int, pd.DataFrame], bayes_table: pd.DataFrame, prior_mean: float
) -> None:
    panels = [(f"Floor: {f}+ ratings", t, "avg_rating") for f, t in floor_tables.items()]
    panels.append((f"Bayesian (m={BAYES_PRIOR_VOTES})", bayes_table, "bayes_score"))

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5), sharex=True)
    for ax, (title, table, col) in zip(axes, panels):
        plot_df = table.iloc[::-1]
        ax.barh(plot_df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True), plot_df[col], color="#3b6ea5")
        ax.set_title(title)
        ax.set_xlim(3.5, 4.7)
        ax.set_xlabel("Average rating (1–5)" if col == "avg_rating" else "Weighted score")
        for y, score, n in zip(range(len(plot_df)), plot_df[col], plot_df["n_ratings"]):
            ax.text(score + 0.01, y, f"{score:.2f}  (n={n})", va="center", fontsize=8)
    fig.suptitle("Best-rated movies: hard floors vs. Bayesian weighting")
    fig.text(
        0.01, 0.01,
        f"Bayesian prior mean C = {prior_mean:.2f} (mean of all ratings). "
        "Votes collected ~1997-1998.",
        fontsize=8, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(CHART_PATH, dpi=150, bbox_inches="tight")
    if plt.get_backend().lower() != "agg":
        plt.show()


def print_table(df: pd.DataFrame, score_col: str) -> None:
    cols = ["rank", "title", score_col, "n_ratings", "pct_five_star"]
    print(
        df[cols].to_string(
            index=False,
            formatters={score_col: "{:.3f}".format, "pct_five_star": "{:.1f}%".format},
        )
    )


def main() -> None:
    ratings = pd.read_csv(DATA_PATH)
    movies = movie_level_stats(ratings)
    movies, prior_mean = add_bayesian_score(movies)

    floor_tables = {floor: top_with_floor(movies, floor) for floor in FLOORS}
    qualified_counts = {floor: int((movies["n_ratings"] >= floor).sum()) for floor in FLOORS}
    bayes_table = top_bayesian(movies)

    payload = top_movies_payload(floor_tables, qualified_counts, bayes_table, prior_mean)
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Movies rated: {len(movies):,}  |  median ratings per movie: {movies['n_ratings'].median():.0f}")
    for floor, table in floor_tables.items():
        print()
        print(f"Top {TOP_N} with at least {floor} ratings ({qualified_counts[floor]} movies qualify)")
        print_table(table, "avg_rating")

    dropped = set(floor_tables[FLOORS[0]]["title"]) - set(floor_tables[FLOORS[-1]]["title"])
    if dropped:
        print()
        print(f"Dropped when floor rises {FLOORS[0]} -> {FLOORS[-1]}: " + "; ".join(sorted(dropped)))

    print()
    print(f"Top {TOP_N} by Bayesian score (m={BAYES_PRIOR_VOTES}, C={prior_mean:.2f}, no floor)")
    print_table(bayes_table, "bayes_score")
    print(f"JSON saved to {JSON_PATH}")

    plot_top_movies(floor_tables, bayes_table, prior_mean)
    print(f"Chart saved to {CHART_PATH}")


if __name__ == "__main__":
    main()
