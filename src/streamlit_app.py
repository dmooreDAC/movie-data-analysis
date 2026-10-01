import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Reuse your existing analysis logic
from movies import DATA_PATH, explode_genres, rated_movie_genres, INVALID_GENRES
from genre_satisfaction import (
    genre_satisfaction,
    satisfaction_payload,
    SMALL_N_THRESHOLD,
)
from ratings_over_time import (
    ratings_by_release_decade,
    ratings_over_time_payload,
    CAVEATS,
)
from top_movies import (
    movie_level_stats as top_movie_level_stats,
    add_bayesian_score,
    top_with_floor,
    top_bayesian,
    top_movies_payload,
    FLOORS,
    TOP_N,
    BAYES_PRIOR_VOTES,
)

st.set_page_config(page_title="Movie Ratings Analysis", layout="wide")
st.title("Movie Ratings Analysis")

# ---------- Load data once ----------
@st.cache_data
def load_ratings() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)

ratings = load_ratings()
st.caption(
    f"{len(ratings):,} ratings · "
    f"{ratings['movie_id'].nunique():,} unique movies"
)

# ---------- Tabs ----------
tab_dist, tab_genre, tab_time, tab_top = st.tabs(
    ["Genre distribution", "Genre satisfaction", "Ratings over time", "Top movies"]
)

# ========== 1. Genre distribution ==========
with tab_dist:
    st.header("Genre mix among uniquely rated movies")
    movies, counts, multi_genre_share = rated_movie_genres(ratings)
    n_titles = len(movies)

    col1, col2 = st.columns(2)
    col1.metric("Unique rated movies", f"{n_titles:,}")
    col2.metric("Have 2+ genres", f"{multi_genre_share:.0%}")

    plot_df = counts.sort_values("n_movies", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df["genre"], plot_df["pct_of_rated_titles"], color="#3b6ea5")
    ax.set_xlabel("% of rated movies tagged with this genre")
    ax.set_xlim(0, plot_df["pct_of_rated_titles"].max() * 1.15)
    for y, pct, n in zip(range(len(plot_df)), plot_df["pct_of_rated_titles"], plot_df["n_movies"]):
        ax.text(pct + 0.4, y, f"{pct:.1f}%  ({n})", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Percents sum > 100% because hybrids count toward each tag. "
        f"{multi_genre_share:.0%} of titles have 2+ genres."
    )

    st.subheader("Table")
    st.dataframe(counts, use_container_width=True)

# ========== 2. Genre satisfaction ==========
with tab_genre:
    st.header("Genre satisfaction: mean of per-movie averages")
    by_genre, overall_movie_mean = genre_satisfaction(ratings)

    col1, col2 = st.columns(2)
    col1.metric("Catalog mean (each movie equal)", f"{overall_movie_mean:.2f} / 5")
    col2.metric("Genres analyzed", len(by_genre))

    plot_df = by_genre.sort_values("avg_rating", ascending=True)
    colors = ["#8aa8c4" if small else "#3b6ea5" for small in plot_df["small_n"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["genre"], plot_df["avg_rating"], color=colors)
    ax.axvline(overall_movie_mean, color="#c45c26", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Average movie rating (1–5)")
    ax.set_xlim(1, 5.35)
    for y, avg, n, small in zip(
        range(len(plot_df)), plot_df["avg_rating"], plot_df["n_movies"], plot_df["small_n"]
    ):
        note = "  | small n" if small else ""
        ax.text(avg + 0.04, y, f"{avg:.1f}  ({n} titles{note})", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Orange line: catalog mean ({overall_movie_mean:.2f}). "
        f"Lighter bars: fewer than {SMALL_N_THRESHOLD} titles."
    )

    st.subheader("Table")
    st.dataframe(
        by_genre,
        use_container_width=True,
        column_config={
            "avg_rating": st.column_config.NumberColumn(format="%.2f"),
            "delta_vs_overall": st.column_config.NumberColumn(format="%+.2f"),
            "small_n": st.column_config.CheckboxColumn(),
        },
    )

    with st.expander("JSON payload"):
        st.json(satisfaction_payload(by_genre, overall_movie_mean))

# ========== 3. Ratings over time ==========
with tab_time:
    st.header("Ratings over time: satisfaction by release decade")
    for caveat in CAVEATS:
        st.warning(caveat)

    by_decade, overall_movie_mean = ratings_by_release_decade(ratings)

    plot_df = by_decade  # already sorted by decade
    colors = ["#8aa8c4" if small else "#3b6ea5" for small in plot_df["small_n"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(plot_df))
    ax.bar(x, plot_df["avg_rating"], color=colors)
    ax.axhline(overall_movie_mean, color="#c45c26", linestyle="--", linewidth=1.5)
    ax.set_xticks(list(x), plot_df["label"])
    ax.set_ylim(1, 5)
    ax.set_xlabel("Movie release decade")
    ax.set_ylabel("Average movie rating (1–5)")
    for i, avg, n, small in zip(x, plot_df["avg_rating"], plot_df["n_movies"], plot_df["small_n"]):
        note = "*" if small else ""
        ax.text(i, avg + 0.06, f"{avg:.1f}{note}\n({n})", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"Orange line: catalog mean ({overall_movie_mean:.2f}). "
        f"* / lighter bars: fewer than {SMALL_N_THRESHOLD} titles."
    )

    st.subheader("Table")
    st.dataframe(by_decade, use_container_width=True)

    with st.expander("JSON payload"):
        st.json(ratings_over_time_payload(by_decade, overall_movie_mean))

# ========== 4. Top movies ==========
with tab_top:
    st.header("Best-rated movies: hard floors vs. Bayesian weighting")

    movies = top_movie_level_stats(ratings)
    movies, prior_mean = add_bayesian_score(movies)

    floor_tables = {floor: top_with_floor(movies, floor) for floor in FLOORS}
    qualified_counts = {floor: int((movies["n_ratings"] >= floor).sum()) for floor in FLOORS}
    bayes_table = top_bayesian(movies)

    st.info(
        f"**Bayesian prior:** C = {prior_mean:.2f} (vote-weighted mean of all ratings), "
        f"m = {BAYES_PRIOR_VOTES} votes. Formula: `(v/(v+m))*R + (m/(v+m))*C`"
    )

    # Three-panel chart
    panels = [(f"Floor: {f}+ ratings", t, "avg_rating") for f, t in floor_tables.items()]
    panels.append((f"Bayesian (m={BAYES_PRIOR_VOTES})", bayes_table, "bayes_score"))

    fig, axes = plt.subplots(1, len(panels), figsize=(16, 5), sharex=True)
    for ax, (title, table, col) in zip(axes, panels):
        plot_df = table.iloc[::-1]
        ax.barh(
            plot_df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True),
            plot_df[col],
            color="#3b6ea5",
        )
        ax.set_title(title)
        ax.set_xlim(3.5, 4.7)
        ax.set_xlabel("Average rating (1–5)" if col == "avg_rating" else "Weighted score")
        for y, score, n in zip(range(len(plot_df)), plot_df[col], plot_df["n_ratings"]):
            ax.text(score + 0.01, y, f"{score:.2f}  (n={n})", va="center", fontsize=8)
    fig.suptitle("Best-rated movies: hard floors vs. Bayesian weighting")
    fig.tight_layout()
    st.pyplot(fig)

    # Show each floor's table
    for floor, table in floor_tables.items():
        st.subheader(f"Top {TOP_N} with ≥ {floor} ratings")
        st.caption(f"{qualified_counts[floor]:,} movies qualify")
        st.dataframe(table, use_container_width=True)

    st.subheader(f"Top {TOP_N} by Bayesian score (no floor)")
    st.dataframe(bayes_table, use_container_width=True)

    with st.expander("JSON payload"):
        st.json(
            top_movies_payload(floor_tables, qualified_counts, bayes_table, prior_mean)
        )