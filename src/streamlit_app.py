import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ---------- Collapsible filter panel ----------
with st.expander("🔎 Filters", expanded=False):
    col_year, col_floor = st.columns(2)

    # Year range
    min_year = int(ratings["year"].min())
    max_year = int(ratings["year"].max())
    year_range = col_year.slider(
        "Release year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1,
        help="Drag either handle to narrow the year window.",
    )

    # Minimum ratings floor
    max_ratings = int(ratings.groupby("movie_id")["rating"].size().max())
    min_ratings = col_floor.slider(
        "Minimum ratings per movie",
        min_value=0,
        max_value=max_ratings,
        value=0,
        step=5,
        help="Drop movies with fewer than this many ratings.",
    )

# ---------- Apply the filter ----------
# Filter movie-level first so we don't drop ratings mid-movie
movie_counts = ratings.groupby("movie_id")["rating"].size().rename("n_ratings")
valid_movie_ids = movie_counts[movie_counts >= min_ratings].index

filtered = ratings[
    (ratings["year"].between(year_range[0], year_range[1]))
    & (ratings["movie_id"].isin(valid_movie_ids))
].copy()

if len(filtered) == 0:
    st.error("No ratings match the current filters. Widen the year range or lower the floor.")
    st.stop()

st.caption(
    f"**Filtered:** {filtered['movie_id'].nunique():,} movies · "
    f"{len(filtered):,} ratings · "
    f"{year_range[0]}–{year_range[1]} · "
    f"≥ {min_ratings} ratings/movie"
)

tab_dist, tab_genre, tab_time, tab_top = st.tabs(
    ["Genre distribution", "Genre satisfaction", "Ratings over time", "Top movies"]
)

# ---------- Tabs ----------
with tab_dist:
    st.header("Genre mix among uniquely rated movies")
    movies, counts, multi_genre_share = rated_movie_genres(ratings)
    n_titles = len(movies)

    col1, col2 = st.columns(2)
    col1.metric("Unique rated movies", f"{n_titles:,}")
    col2.metric("Have 2+ genres", f"{multi_genre_share:.0%}")

    plot_df = counts.sort_values("n_movies", ascending=True)
    fig = px.bar(
        plot_df,
        x="pct_of_rated_titles",
        y="genre",
        orientation="h",
        color="pct_of_rated_titles",
        color_continuous_scale="Blues",
        custom_data=["n_movies"],
        height=600,
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x:.1f}% of rated titles<br>%{customdata[0]} movies<extra></extra>"
    )
    fig.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="% of rated movies tagged",
        yaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Percents sum > 100% because hybrids count toward each tag. "
        f"{multi_genre_share:.0%} of titles have 2+ genres."
    )

# ========== 2. Genre satisfaction ==========
with tab_genre:
    st.header("Genre satisfaction: mean of per-movie averages")
    by_genre, overall_movie_mean = genre_satisfaction(ratings)

    col1, col2 = st.columns(2)
    col1.metric("Catalog mean (each movie equal)", f"{overall_movie_mean:.2f} / 5")
    col2.metric("Genres analyzed", len(by_genre))

    plot_df = by_genre.sort_values("avg_rating", ascending=True).copy()
    plot_df["reliability"] = plot_df["small_n"].map({True: "small n (< 30)", False: "reliable"})

    fig = px.bar(
        plot_df,
        x="avg_rating",
        y="genre",
        orientation="h",
        color="reliability",
        color_discrete_map={"reliable": "#3b6ea5", "small n (< 30)": "#8aa8c4"},
        custom_data=["n_movies", "n_ratings", "delta_vs_overall", "small_n"],
        height=650,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg rating: %{x:.2f}<br>"
            "Movies: %{customdata[0]}<br>"
            "Ratings: %{customdata[1]:,}<br>"
            "Δ vs catalog: %{customdata[2]:+.2f}"
            "<extra></extra>"
        )
    )
    fig.add_vline(
        x=overall_movie_mean,
        line_dash="dash",
        line_color="#c45c26",
        line_width=2,
        annotation_text=f"catalog mean {overall_movie_mean:.2f}",
        annotation_position="top",
        annotation_font_color="#c45c26",
    )
    fig.update_layout(
        xaxis=dict(range=[1, 5], title="Average movie rating (1–5)"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Lighter bars: fewer than {SMALL_N_THRESHOLD} titles. "
        "Hybrids count toward each genre."
    )


# ========== 3. Ratings over time ==========
with tab_time:
    st.header("Ratings over time: satisfaction by release decade")
    for caveat in CAVEATS:
        st.warning(caveat)

    by_decade, overall_movie_mean = ratings_by_release_decade(filtered)

    if by_decade.empty:
        st.warning("No decades in the selected year range. Widen the filter above.")
        st.stop()

    plot_df = by_decade.copy()
    plot_df["reliability"] = plot_df["small_n"].map(
        {True: "small n (< 30)", False: "reliable"}
    )
    plot_df["marker_size"] = (plot_df["n_movies"] ** 0.5) * 4

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df["label"],
            y=plot_df["avg_rating"],
            mode="lines+markers",
            line=dict(color="#3b6ea5", width=3),
            marker=dict(
                size=plot_df["marker_size"],
                color=["#8aa8c4" if s else "#3b6ea5" for s in plot_df["small_n"]],
                line=dict(color="white", width=1.5),
            ),
            customdata=plot_df[["n_movies", "n_ratings", "delta_vs_overall"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Avg rating: %{y:.2f}<br>"
                "Movies: %{customdata[0]}<br>"
                "Ratings: %{customdata[1]:,}<br>"
                "Δ vs catalog: %{customdata[2]:+.2f}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    for x, y, n, small in zip(
        plot_df["label"], plot_df["avg_rating"], plot_df["n_movies"], plot_df["small_n"]
    ):
        note = "*" if small else ""
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{y:.2f}{note}",
            showarrow=False,
            yshift=14,
            font=dict(size=10, color="#3b6ea5"),
        )

    fig.add_hline(
        y=overall_movie_mean,
        line_dash="dash",
        line_color="#c45c26",
        line_width=2,
        annotation_text=f"catalog mean {overall_movie_mean:.2f}",
        annotation_position="bottom right",
        annotation_font_color="#c45c26",
    )

    # KEY: constrain x-axis to only the decades present in the filtered data
    fig.update_layout(
        yaxis=dict(range=[2.5, 4.0], title="Average movie rating (1–5)"),
        xaxis=dict(
            title="Movie release decade",
            categoryorder="array",
            categoryarray=list(plot_df["label"]),  # only the surviving decades
            type="category",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        hovermode="x unified",
        transition=dict(duration=300),  # smooth re-render on filter change
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Showing {len(plot_df)} decade{'s' if len(plot_df) != 1 else ''} "
        f"({plot_df['label'].iloc[0]}–{plot_df['label'].iloc[-1]}). "
        "Marker size reflects the number of titles. "
        "* / lighter markers: fewer than 30 titles."
    )

# ========== 4. Top movies ==========
with tab_top:
    st.header("Best-rated movies: raw average vs. Bayesian weighting")

    # Per-movie stats on the filtered ratings
    movies = top_movie_level_stats(filtered)
    movies, prior_mean = add_bayesian_score(movies)

    # The filter slider is the floor — no hardcoded FLOORS
    floor = min_ratings  # comes from the expander above
    floor_table = top_with_floor(movies, floor, top_n=TOP_N)
    qualified_count = int((movies["n_ratings"] >= floor).sum())

    bayes_table = top_bayesian(movies, top_n=TOP_N)

    st.info(
        f"**Floor:** ≥ {floor} ratings → {qualified_count:,} movies qualify  \n"
        f"**Bayesian prior:** C = {prior_mean:.2f} (vote-weighted mean of all ratings), "
        f"m = {BAYES_PRIOR_VOTES} votes. "
        f"Formula: `(v/(v+m))*R + (m/(v+m))*C`"
    )

    # Two panels: raw average with floor, and Bayesian score with the same floor
    panels = [
        (f"Raw average (≥ {floor} ratings)", floor_table, "avg_rating"),
        (f"Bayesian (m={BAYES_PRIOR_VOTES}, ≥ {floor} ratings)", bayes_table, "bayes_score"),
    ]

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_xaxes=True,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.12,
    )

    for i, (title, table, col) in enumerate(panels, start=1):
        plot_df = table.iloc[::-1].copy()
        plot_df["short_title"] = plot_df["title"].str.replace(
            r"\s*\(\d{4}\)$", "", regex=True
        )
        fig.add_trace(
            go.Bar(
                x=plot_df[col],
                y=plot_df["short_title"],
                orientation="h",
                marker_color="#3b6ea5",
                customdata=plot_df[
                    ["n_ratings", "avg_rating", "bayes_score", "pct_five_star"]
                ].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Raw avg: %{customdata[1]:.2f}<br>"
                    "Bayes score: %{customdata[2]:.3f}<br>"
                    "Ratings: %{customdata[0]}<br>"
                    "5★ share: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=i,
        )

    fig.update_xaxes(range=[3.5, 4.7])
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Both panels respect the filter above. Raw average is the plain per-movie mean; "
        "Bayesian shrinks thin-sample movies toward the catalog prior."
    )

