# 🏒 Moneyball Phil — NHL EV Simulator (FINAL COMPLETE VERSION)
# Tabs: Team Bets · Player Props
# Engines: Poisson (Teams) + Poisson/Binomial (Player Props)
# Includes: Dynamic puck line favorite selection, EV + Tier + Visuals

import math
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏒 Moneyball Phil — NHL EV Simulator", layout="wide")
st.title("🏒 Moneyball Phil — NHL EV Simulator")

# =========================
# ===== Session State =====
# =========================
def _init_state():
    st.session_state.setdefault("bets_board", [])
    st.session_state.setdefault("parlay_current", [])
    st.session_state.setdefault("parlay_board", [])
    st.session_state.setdefault("_leg_id", 1)
_init_state()

# =========================
# ===== Helper Utils ======
# =========================
def american_to_implied(odds: float) -> float:
    try:
        odds = float(odds)
    except:
        return None
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0) if odds >= 0 else 1.0 + (100.0 / abs(odds))

def decimal_to_american(dec: float) -> float:
    dec = float(dec)
    if dec <= 1.0:
        return 0.0
    if dec >= 2.0:
        return (dec - 1.0) * 100.0
    return -100.0 / (dec - 1.0)

def ev_percent(true_prob: float, american_odds: float) -> float:
    if true_prob is None or american_odds is None:
        return None
    true_prob = max(0.0, min(1.0, float(true_prob)))
    american_odds = float(american_odds)
    profit = american_odds / 100.0 if american_odds >= 0 else 100.0 / abs(american_odds)
    ev = true_prob * profit - (1.0 - true_prob)
    return ev * 100.0

def ev_percent_from_decimal(true_prob: float, dec_odds: float) -> float:
    true_prob = max(0.0, min(1.0, float(true_prob)))
    profit = float(dec_odds) - 1.0
    ev = true_prob * profit - (1.0 - true_prob)
    return ev * 100.0

def tier_from_prob(p: float) -> str:
    if p is None:
        return "—"
    if p > 0.80:
        return "Elite"
    if p >= 0.70:
        return "Strong"
    if p >= 0.60:
        return "Moderate"
    return "Risky"

def poisson_geq(lam: float, k: int) -> float:
    lam = max(1e-9, float(lam))
    k = max(0, int(math.floor(k)))
    s = 0.0
    for i in range(0, k):
        s += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return 1.0 - s

# =========================
# ===== Team Engines ======
# =========================
def expected_goals_pair(xgf_home, xga_home, xgf_away, xga_away):
    lam_home = (xgf_home + xga_away) / 2.0
    lam_away = (xgf_away + xga_home) / 2.0
    return max(0.05, lam_home), max(0.05, lam_away)

def simulate_matchups(lam_home, lam_away, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(lam_home, size=n)
    away_goals = rng.poisson(lam_away, size=n)
    return home_goals, away_goals

def ml_pw_over_under_metrics(lam_home, lam_away, total_line, n=10000, seed=42):
    hg, ag = simulate_matchups(lam_home, lam_away, n=n, seed=seed)
    return {
        "home_win": np.mean(hg > ag),
        "away_win": np.mean(ag > hg),
        "home_cover_-1.5": np.mean((hg - ag) >= 2),
        "away_cover_+1.5": np.mean((ag - hg) > -2),
        "over": np.mean((hg + ag) > total_line),
        "under": np.mean((hg + ag) < total_line),
    }

# =========================
# ===== Player Engines =====
# =========================
def weighted_rate(season_avg, recent_avg, weight_recent=0.7):
    w = max(0.0, min(1.0, float(weight_recent)))
    return (w * float(recent_avg)) + ((1 - w) * float(season_avg))

def defense_adjust(rate, opp_allowed, league_avg=None):
    if opp_allowed is None:
        return max(0.0, rate)
    if league_avg and league_avg > 0:
        return max(0.0, rate * (opp_allowed / league_avg))
    return max(0.0, rate * (1 + (opp_allowed - rate) * 0.10 / max(0.1, rate)))

def prob_point_yes(rate_points_per_game):
    lam = max(0.0, float(rate_points_per_game))
    return 1.0 - math.exp(-lam)

def prob_count_over_poisson(rate_per_game, line):
    k = math.ceil(line)
    return poisson_geq(max(rate_per_game, 1e-6), k)

# =========================
# ===== Matplotlib UI =====
# =========================
def plot_bar_true_vs_implied(true_p, implied_p, title):
    fig, ax = plt.subplots()
    cats = ["True", "Implied"]
    vals = [100.0 * max(0.0, min(1.0, true_p)), 100.0 * max(0.0, min(1.0, implied_p))]
    ax.bar(cats, vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title)
    st.pyplot(fig)

# =========================
# ========= Layout =========
# =========================
tab_team, tab_player = st.tabs(["Team Bets", "Player Props"])

# -------------------------
# TEAM BETS
# -------------------------
with tab_team:
    st.subheader("Team Bets — Moneyline · Puck Line · Totals")

    colL, colR = st.columns(2)
    with colL:
        home_team = st.text_input("Home Team", value="")
        away_team = st.text_input("Away Team", value="")
        xgf_home = st.number_input("Home xGF (per game)", value=0.0, step=0.1, min_value=0.0)
        xga_home = st.number_input("Home xGA (per game)", value=0.0, step=0.1, min_value=0.0)
        xgf_away = st.number_input("Away xGF (per game)", value=0.0, step=0.1, min_value=0.0)
        xga_away = st.number_input("Away xGA (per game)", value=0.0, step=0.1, min_value=0.0)
        sims = 20000

    with colR:
        st.markdown("**Sportsbook Lines** (enter your current prices)")
        ml_home = st.number_input(f"Moneyline — {home_team or 'Home Team'}", value=0, step=1, key="ml_home")
        ml_away = st.number_input(f"Moneyline — {away_team or 'Away Team'}", value=0, step=1, key="ml_away")

        favorite_team = st.selectbox(
            "Which team is the favorite for the puck line?",
            options=[home_team or "Home", away_team or "Away"],
            index=0,
            key="favorite_select"
        )
        if favorite_team == (home_team or "Home"):
            pl_fav_label = f"Puck Line {home_team or 'Home'} -1.5 (odds)"
            pl_dog_label = f"Puck Line {away_team or 'Away'} +1.5 (odds)"
        else:
            pl_fav_label = f"Puck Line {away_team or 'Away'} -1.5 (odds)"
            pl_dog_label = f"Puck Line {home_team or 'Home'} +1.5 (odds)"

        pl_fav = st.number_input(pl_fav_label, value=0, step=1, key="pl_fav")
        pl_dog = st.number_input(pl_dog_label, value=0, step=1, key="pl_dog")
        total_line = st.number_input("Total (O/U) line", value=0.0, step=0.5, key="total_line")
        ou_over = st.number_input("Over Odds", value=0, step=1, key="ou_over")
        ou_under = st.number_input("Under Odds", value=0, step=1, key="ou_under")

    if st.button("Run Team Simulation"):
        lam_h, lam_a = expected_goals_pair(xgf_home, xga_home, xgf_away, xga_away)
        metrics = ml_pw_over_under_metrics(lam_h, lam_a, total_line, n=int(sims))

        p_home, p_away = metrics["home_win"], metrics["away_win"]
        p_fav_pl = metrics["home_cover_-1.5"] if favorite_team == (home_team or "Home") else metrics["away_cover_+1.5"]
        p_dog_pl = metrics["away_cover_+1.5"] if favorite_team == (home_team or "Home") else metrics["home_cover_-1.5"]
        p_over, p_under = metrics["over"], metrics["under"]

        imp_home, imp_away = american_to_implied(ml_home), american_to_implied(ml_away)
        imp_fav_pl, imp_dog_pl = american_to_implied(pl_fav), american_to_implied(pl_dog)
        imp_over, imp_under = american_to_implied(ou_over), american_to_implied(ou_under)

        ev_home, ev_away = ev_percent(p_home, ml_home), ev_percent(p_away, ml_away)
        ev_fav_pl, ev_dog_pl = ev_percent(p_fav_pl, pl_fav), ev_percent(p_dog_pl, pl_dog)
        ev_over, ev_under = ev_percent(p_over, ou_over), ev_percent(p_under, ou_under)

        st.markdown("### Results (True %, Implied %, EV %, Tier)")
        rows = [
            {"Market": f"ML — {home_team or 'Home'}", "True %": p_home*100, "Implied %": imp_home*100 if imp_home else None, "EV %": ev_home, "Tier": tier_from_prob(p_home)},
            {"Market": f"ML — {away_team or 'Away'}", "True %": p_away*100, "Implied %": imp_away*100 if imp_away else None, "EV %": ev_away, "Tier": tier_from_prob(p_away)},
            {"Market": pl_fav_label, "True %": p_fav_pl*100, "Implied %": imp_fav_pl*100 if imp_fav_pl else None, "EV %": ev_fav_pl, "Tier": tier_from_prob(p_fav_pl)},
            {"Market": pl_dog_label, "True %": p_dog_pl*100, "Implied %": imp_dog_pl*100 if imp_dog_pl else None, "EV %": ev_dog_pl, "Tier": tier_from_prob(p_dog_pl)},
            {"Market": f"Over {total_line}", "True %": p_over*100, "Implied %": imp_over*100 if imp_over else None, "EV %": ev_over, "Tier": tier_from_prob(p_over)},
            {"Market": f"Under {total_line}", "True %": p_under*100, "Implied %": imp_under*100 if imp_under else None, "EV %": ev_under, "Tier": tier_from_prob(p_under)},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Visuals — True vs Implied Probability")
        colA, colB, colC = st.columns(3)
        with colA:
            plot_bar_true_vs_implied(p_home, imp_home, f"ML — {home_team or 'Home'}")
            plot_bar_true_vs_implied(p_away, imp_away, f"ML — {away_team or 'Away'}")
        with colB:
            plot_bar_true_vs_implied(p_fav_pl, imp_fav_pl, pl_fav_label)
            plot_bar_true_vs_implied(p_dog_pl, imp_dog_pl, pl_dog_label)
        with colC:
            plot_bar_true_vs_implied(p_over, imp_over, f"Over {total_line}")
            plot_bar_true_vs_implied(p_under, imp_under, f"Under {total_line}")

# -------------------------
# PLAYER PROPS
# -------------------------
with tab_player:
    st.subheader("Player Props — Points · Goals · Assists · SOG")

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player Name", value="")
        prop_type = st.selectbox("Prop Type", ["To Record a Point (Yes)", "Goals Over", "Assists Over", "Shots on Goal Over"])
        line = st.number_input("Prop Line", value=0.0, step=0.5)
        season_avg = st.number_input("Season Avg", value=0.0, step=0.05, min_value=0.0)
        recent_avg = st.number_input("Last 7 Avg", value=0.0, step=0.05, min_value=0.0)
        weight_recent = st.slider("Recent Weight", 0.0, 1.0, 0.7, 0.05)

    with col2:
        opp_allowed = st.number_input("Opponent Allowed", value=0.0, step=0.05, min_value=0.0)
        league_avg = st.number_input("League Avg (optional)", value=0.0, step=0.05, min_value=0.0)
        odds_over = st.number_input("Over Odds", value=0, step=1, key="player_over_odds")
        odds_under = st.number_input("Under Odds", value=0, step=1, key="player_under_odds")

    if st.button("Compute Player Prop"):
        base_rate = weighted_rate(season_avg, recent_avg, weight_recent)
        adj_rate = defense_adjust(base_rate, opp_allowed, league_avg if league_avg > 0 else None)
        if prop_type == "To Record a Point (Yes)":
            true_over = prob_point_yes(adj_rate)
        else:
            true_over = prob_count_over_poisson(adj_rate, line)
        true_under = 1.0 - true_over

        imp_over, imp_under = american_to_implied(odds_over), american_to_implied(odds_under)
        ev_over, ev_under = ev_percent(true_over, odds_over), ev_percent(true_under, odds_under)

        st.dataframe(pd.DataFrame([
            {"Market": f"{player_name} — {prop_type}", "Side": "Over/Yes", "True %": true_over*100, "Implied %": imp_over*100 if imp_over else None, "EV %": ev_over, "Tier": tier_from_prob(true_over)},
            {"Market": f"{player_name} — {prop_type}", "Side": "Under/No", "True %": true_under*100, "Implied %": imp_under*100 if imp_under else None, "EV %": ev_under, "Tier": tier_from_prob(true_under)},
        ]), use_container_width=True)

        st.markdown("#### Visuals — True vs Implied Probability")
        plot_bar_true_vs_implied(true_over, imp_over, f"{player_name} — Over/Yes")
        plot_bar_true_vs_implied(true_under, imp_under, f"{player_name} — Under/No")
# =========================
# ===== Boards & Parlay ====
# =========================
st.markdown("---")
st.header("📋 Bet Boards & Parlay Builder")

# Current Bets Board
st.subheader("Saved Single Bets")
if st.session_state["bets_board"]:
    st.dataframe(pd.DataFrame(st.session_state["bets_board"]), use_container_width=True)
    if st.button("Clear Single-Bet Board"):
        st.session_state["bets_board"] = []
else:
    st.caption("No single bets saved yet.")

# Current Parlay Working Area
st.subheader("Current Parlay (Working)")
if st.session_state["parlay_current"]:
    df_parlay = pd.DataFrame(st.session_state["parlay_current"])
    st.dataframe(df_parlay, use_container_width=True)

    # Calculate combined parlay metrics
    def parlay_metrics(legs: List[Dict]):
        if not legs:
            return None
        true_comb = 1.0
        dec_prod = 1.0
        for lg in legs:
            true_comb *= max(0.0, min(1.0, float(lg["True"])))
            dec_prod *= float(lg["Decimal"])
        implied_comb = 1.0 / dec_prod
        am_comb = decimal_to_american(dec_prod)
        ev_comb = ev_percent_from_decimal(true_comb, dec_prod)
        return {
            "True %": true_comb * 100.0,
            "Implied %": implied_comb * 100.0,
            "Decimal": dec_prod,
            "Odds (Am)": am_comb,
            "EV %": ev_comb,
        }

    met = parlay_metrics(st.session_state["parlay_current"])
    if met:
        st.write(f"**Combined True %:** {met['True %']:.2f}% | **Implied %:** {met['Implied %']:.2f}% | "
                 f"**Odds (Am):** {met['Odds (Am)']:.0f} | **EV:** {met['EV %']:.2f}%")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Parlay to Board"):
            st.session_state["parlay_board"].append({
                "Legs": [l["Market"] for l in st.session_state["parlay_current"]],
                "Odds (Am)": met["Odds (Am)"],
                "Decimal": met["Decimal"],
                "True %": met["True %"],
                "Implied %": met["Implied %"],
                "EV %": met["EV %"],
            })
            st.session_state["parlay_current"] = []
    with c2:
        if st.button("Clear Current Parlay"):
            st.session_state["parlay_current"] = []
else:
    st.caption("No legs in the current parlay yet. Add legs from Team Bets or Player Props above.")

# Saved Parlays Board
st.subheader("Saved Parlays Board")
if st.session_state["parlay_board"]:
    formatted = []
    for i, p in enumerate(st.session_state["parlay_board"], start=1):
        formatted.append({
            "#": i,
            "Legs": " | ".join(p["Legs"]),
            "Odds (Am)": round(p["Odds (Am)"], 0),
            "True %": round(p["True %"], 2),
            "Implied %": round(p["Implied %"], 2),
            "EV %": round(p["EV %"], 2),
        })
    st.dataframe(pd.DataFrame(formatted), use_container_width=True)
    if st.button("Clear Saved Parlays"):
        st.session_state["parlay_board"] = []
else:
    st.caption("No parlays saved yet.")
