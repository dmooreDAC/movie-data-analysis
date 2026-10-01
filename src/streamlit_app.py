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

    with st.expander("See raw numbers"):
        st.dataframe(counts, use_container_width=True)

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

    with st.expander("See raw numbers"):
        st.dataframe(by_genre, use_container_width=True)

    with st.expander("JSON payload"):
        st.json(satisfaction_payload(by_genre, overall_movie_mean))

# ========== 3. Ratings over time ==========
with tab_time:
    st.header("Ratings over time: satisfaction by release decade")
    for caveat in CAVEATS:
        st.warning(caveat)

    by_decade, overall_movie_mean = ratings_by_release_decade(ratings)

    plot_df = by_decade.copy()
    plot_df["reliability"] = plot_df["small_n"].map({True: "small n (< 30)", False: "reliable"})

    fig = px.bar(
        plot_df,
        x="label",
        y="avg_rating",
        color="reliability",
        color_discrete_map={"reliable": "#3b6ea5", "small n (< 30)": "#8aa8c4"},
        custom_data=["n_movies", "n_ratings", "delta_vs_overall", "small_n"],
        height=500,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Avg rating: %{y:.2f}<br>"
            "Movies: %{customdata[0]}<br>"
            "Ratings: %{customdata[1]:,}<br>"
            "Δ vs catalog: %{customdata[2]:+.2f}"
            "<extra></extra>"
        )
    )
    fig.add_hline(
        y=overall_movie_mean,
        line_dash="dash",
        line_color="#c45c26",
        line_width=2,
        annotation_text=f"catalog mean {overall_movie_mean:.2f}",
        annotation_position="top left",
        annotation_font_color="#c45c26",
    )
    fig.update_layout(
        yaxis=dict(range=[1, 5], title="Average movie rating (1–5)"),
        xaxis=dict(title="Movie release decade"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The 1990s bar dominates because the catalog is heavily weighted toward "
        "recently released films. Earlier decades are a survivor shelf."
    )

    with st.expander("See raw numbers"):
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

    panels = [(f"Floor: {f}+ ratings", t, "avg_rating") for f, t in floor_tables.items()]
    panels.append((f"Bayesian (m={BAYES_PRIOR_VOTES})", bayes_table, "bayes_score"))

    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_xaxes=True,
        subplot_titles=[p[0] for p in panels],
        horizontal_spacing=0.08,
    )
    for i, (title, table, col) in enumerate(panels, start=1):
        plot_df = table.iloc[::-1].copy()
        plot_df["short_title"] = plot_df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        fig.add_trace(
            go.Bar(
                x=plot_df[col],
                y=plot_df["short_title"],
                orientation="h",
                marker_color="#3b6ea5",
                customdata=plot_df[["n_ratings", "avg_rating", "bayes_score", "pct_five_star"]].values,
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
        "Same films, three methods. Films in the left panel but not the others "
        "(e.g., *A Close Shave*, *Wallace & Gromit*) have high averages but thin samples."
    )

    with st.expander("See raw numbers"):
        for floor, table in floor_tables.items():
            st.subheader(f"Top {TOP_N} with ≥ {floor} ratings ({qualified_counts[floor]:,} qualify)")
            st.dataframe(table, use_container_width=True)
        st.subheader(f"Top {TOP_N} by Bayesian score (no floor)")
        st.dataframe(bayes_table, use_container_width=True)

    with st.expander("JSON payload"):
        st.json(
            top_movies_payload(floor_tables, qualified_counts, bayes_table, prior_mean)
        )