# 🏒 Moneyball Phil — NHL EV Simulator (Standalone)
# Two tabs: Team Bets (Moneyline / Puck Line / Totals) and Player Props
# Includes Poisson (teams) and Binomial/Poisson (player props) engines + matplotlib bar charts
# Streamlit-only dependencies: streamlit, numpy, pandas, matplotlib

import math
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="🏒 Moneyball Phil — NHL EV Simulator", layout="wide")
st.title("🏒 Moneyball Phil — NHL EV Simulator")

# =========================
# ===== Helper Utils ======
# =========================

def american_to_implied(odds: float) -> float:
    """Return implied probability from American odds.
    User preference (Model Set Context):
    - negative odds: |odds| / (|odds| + 100)
    - positive odds: 100 / (odds + 100)
    """
    if odds is None:
        return None
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(p: float) -> float:
    """(Optional) Convert implied probability to fair American odds (not used in UI)."""
    p = max(1e-9, min(0.999999, p))
    if p >= 0.5:
        return - (p * 100) / (1 - p)
    return (100 * (1 - p)) / p


def ev_percent(true_prob: float, american_odds: float) -> float:
    """Expected value per $1 stake, expressed as a percentage.
    Profit on win for +odds = odds/100; for -odds = 100/|odds|.
    EV = true*payout - (1-true)*1. Return as %.
    """
    if true_prob is None or american_odds is None:
        return None
    true_prob = max(0.0, min(1.0, float(true_prob)))
    american_odds = float(american_odds)
    if american_odds >= 0:
        profit = american_odds / 100.0
    else:
        profit = 100.0 / abs(american_odds)
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
    """P(X >= k) for Poisson(lambda)."""
    lam = max(1e-9, float(lam))
    k = max(0, int(math.floor(k)))
    # 1 - CDF(k-1)
    s = 0.0
    for i in range(0, k):
        s += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return 1.0 - s


# =========================
# ===== Team Engines ======
# =========================

def expected_goals_pair(xgf_home: float, xga_home: float, xgf_away: float, xga_away: float) -> Tuple[float, float]:
    """Simple xG fusion for home/away expected scoring rates."""
    lam_home = (float(xgf_home) + float(xga_away)) / 2.0
    lam_away = (float(xgf_away) + float(xga_home)) / 2.0
    return max(0.05, lam_home), max(0.05, lam_away)


def simulate_matchups(lam_home: float, lam_away: float, n: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    home_goals = rng.poisson(lam_home, size=n)
    away_goals = rng.poisson(lam_away, size=n)
    return home_goals, away_goals


def ml_pw_over_under_metrics(lam_home: float, lam_away: float, total_line: float, n: int = 10000, seed: int = 42):
    hg, ag = simulate_matchups(lam_home, lam_away, n=n, seed=seed)
    home_win = np.mean(hg > ag)
    away_win = np.mean(ag > hg)
    draw_reg = np.mean(hg == ag)  # useful for 3-way markets later

    # Puck line cover: favorite -1.5 (home) and dog +1.5 (away)
    home_cover_m15 = np.mean((hg - ag) >= 2)
    away_cover_p15 = np.mean((ag - hg) > -2)  # i.e., away loses by 0 or 1, or wins

    totals_over = np.mean((hg + ag) > total_line)
    totals_under = np.mean((hg + ag) < total_line)
    totals_push = 1.0 - totals_over - totals_under  # if line happens to be integer (rare)

    return {
        "home_win": home_win,
        "away_win": away_win,
        "draw_reg": draw_reg,
        "home_cover_-1.5": home_cover_m15,
        "away_cover_+1.5": away_cover_p15,
        "over": totals_over,
        "under": totals_under,
        "push": totals_push,
    }


# ==============================
# ===== Player Prop Engines =====
# ==============================

def weighted_rate(season_avg: float, recent_avg: float, weight_recent: float = 0.7) -> float:
    w = max(0.0, min(1.0, float(weight_recent)))
    return (w * float(recent_avg)) + ((1 - w) * float(season_avg))


def defense_adjust(rate: float, opp_allowed: float, league_avg: float = None) -> float:
    """Adjust player's rate by opponent allowed. If league_avg provided, scale multiplicatively."""
    r = float(rate)
    if opp_allowed is None:
        return max(0.0, r)
    oa = max(0.0, float(opp_allowed))
    if league_avg is None or league_avg <= 0:
        # conservative: average shift of +/− 10% around rate
        factor = 1.0 + (oa - r) * 0.10 / max(0.1, r)
        return max(0.0, r * factor)
    # scale by opponent vs league
    factor = oa / float(league_avg)
    return max(0.0, r * factor)


def prob_point_yes(rate_points_per_game: float) -> float:
    """Approximate P(at least one point) via Poisson with mean = points per game rate."""
    lam = max(0.0, float(rate_points_per_game))
    return 1.0 - math.exp(-lam)


def prob_count_over_poisson(rate_per_game: float, line: float) -> float:
    """P(X >= ceil(line+small)) using Poisson, for goals/assists/SOG type integers."""
    k = math.floor(line + 1e-9)
    # For lines like 0.5, k becomes 0, so we need >=1. Adjust:
    if abs(line - (k + 0.5)) < 0.49:
        k = math.ceil(line)  # e.g., 0.5 -> 1, 1.5 -> 2
    else:
        k = math.ceil(line)
    return poisson_geq(max(rate_per_game, 1e-6), k)


# =========================
# ===== Matplotlib UI =====
# =========================

def plot_bar_true_vs_implied(true_p: float, implied_p: float, title: str):
    fig, ax = plt.subplots()
    cats = ["True", "Implied"]
    vals = [100.0 * max(0.0, min(1.0, true_p)), 100.0 * max(0.0, min(1.0, implied_p))]
    ax.bar(cats, vals)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Probability (%)")
    ax.set_title(title)
    st.pyplot(fig)


# =========================
# ======== Layout =========
# =========================

tab_team, tab_player = st.tabs(["Team Bets", "Player Props"])

# -------------------------
# Team Bets Tab
# -------------------------
with tab_team:
    st.subheader("Team Bets — Moneyline · Puck Line · Totals")

    colL, colR = st.columns(2)
    with colL:
        home_team = st.text_input("Home Team", value="Maple Leafs")
        away_team = st.text_input("Away Team", value="Canadiens")
        xgf_home = st.number_input("Home xGF (per game)", value=3.2, step=0.1, min_value=0.0)
        xga_home = st.number_input("Home xGA (per game)", value=2.8, step=0.1, min_value=0.0)
        xgf_away = st.number_input("Away xGF (per game)", value=2.9, step=0.1, min_value=0.0)
        xga_away = st.number_input("Away xGA (per game)", value=3.1, step=0.1, min_value=0.0)
        sims = st.number_input("Simulations (Poisson)", value=20000, step=1000, min_value=5000, help="Use 10k–100k for stability. 20k default.")

    with colR:
        st.markdown("**Sportsbook Lines** (enter your current prices)")
        ml_home = st.number_input(f"Moneyline — {home_team}", value=-135, step=1)
        ml_away = st.number_input(f"Moneyline — {away_team}", value=+115, step=1)
        pl_home = st.number_input(f"Puck Line {home_team} -1.5 (odds)", value=+150, step=1)
        pl_away = st.number_input(f"Puck Line {away_team} +1.5 (odds)", value=-170, step=1)
        total_line = st.number_input("Total (O/U) line", value=6.5, step=0.5)
        ou_over = st.number_input("Over Odds", value=+105, step=1)
        ou_under = st.number_input("Under Odds", value=-125, step=1)

    run_team = st.button("Run Team Simulation")

    if run_team:
        lam_h, lam_a = expected_goals_pair(xgf_home, xga_home, xgf_away, xga_away)
        metrics = ml_pw_over_under_metrics(lam_h, lam_a, total_line, n=int(sims))

        # True probabilities
        p_home = metrics["home_win"]
        p_away = metrics["away_win"]
        p_home_pl = metrics["home_cover_-1.5"]
        p_away_pl = metrics["away_cover_+1.5"]
        p_over = metrics["over"]
        p_under = metrics["under"]

        # Implied from odds
        imp_home = american_to_implied(ml_home)
        imp_away = american_to_implied(ml_away)
        imp_home_pl = american_to_implied(pl_home)
        imp_away_pl = american_to_implied(pl_away)
        imp_over = american_to_implied(ou_over)
        imp_under = american_to_implied(ou_under)

        # EV calculations
        ev_home = ev_percent(p_home, ml_home)
        ev_away = ev_percent(p_away, ml_away)
        ev_home_pl = ev_percent(p_home_pl, pl_home)
        ev_away_pl = ev_percent(p_away_pl, pl_away)
        ev_over = ev_percent(p_over, ou_over)
        ev_under = ev_percent(p_under, ou_under)

        # Display tables
        st.markdown("### Results (True %, Implied %, EV %, Tier)")
        team_rows = [
            {
                "Market": f"ML — {home_team}",
                "True %": round(p_home * 100, 2),
                "Implied %": round(imp_home * 100, 2) if imp_home is not None else None,
                "EV %": round(ev_home, 2) if ev_home is not None else None,
                "Tier": tier_from_prob(p_home),
            },
            {
                "Market": f"ML — {away_team}",
                "True %": round(p_away * 100, 2),
                "Implied %": round(imp_away * 100, 2) if imp_away is not None else None,
                "EV %": round(ev_away, 2) if ev_away is not None else None,
                "Tier": tier_from_prob(p_away),
            },
            {
                "Market": f"Puck Line — {home_team} -1.5",
                "True %": round(p_home_pl * 100, 2),
                "Implied %": round(imp_home_pl * 100, 2) if imp_home_pl is not None else None,
                "EV %": round(ev_home_pl, 2) if ev_home_pl is not None else None,
                "Tier": tier_from_prob(p_home_pl),
            },
            {
                "Market": f"Puck Line — {away_team} +1.5",
                "True %": round(p_away_pl * 100, 2),
                "Implied %": round(imp_away_pl * 100, 2) if imp_away_pl is not None else None,
                "EV %": round(ev_away_pl, 2) if ev_away_pl is not None else None,
                "Tier": tier_from_prob(p_away_pl),
            },
            {
                "Market": f"Total Over {total_line}",
                "True %": round(p_over * 100, 2),
                "Implied %": round(imp_over * 100, 2) if imp_over is not None else None,
                "EV %": round(ev_over, 2) if ev_over is not None else None,
                "Tier": tier_from_prob(p_over),
            },
            {
                "Market": f"Total Under {total_line}",
                "True %": round(p_under * 100, 2),
                "Implied %": round(imp_under * 100, 2) if imp_under is not None else None,
                "EV %": round(ev_under, 2) if ev_under is not None else None,
                "Tier": tier_from_prob(p_under),
            },
        ]
        df_team = pd.DataFrame(team_rows)
        st.dataframe(df_team, use_container_width=True)

        # Bar charts (one chart per market as requested style)
        st.markdown("#### Visuals — True vs Implied Probability")
        colA, colB, colC = st.columns(3)
        with colA:
            plot_bar_true_vs_implied(p_home, imp_home, f"ML — {home_team}")
            plot_bar_true_vs_implied(p_away, imp_away, f"ML — {away_team}")
        with colB:
            plot_bar_true_vs_implied(p_home_pl, imp_home_pl, f"PL {home_team} -1.5")
            plot_bar_true_vs_implied(p_away_pl, imp_away_pl, f"PL {away_team} +1.5")
        with colC:
            plot_bar_true_vs_implied(p_over, imp_over, f"Over {total_line}")
            plot_bar_true_vs_implied(p_under, imp_under, f"Under {total_line}")

        # Quick best EV summary
        ev_map = {
            f"ML — {home_team}": ev_home,
            f"ML — {away_team}": ev_away,
            f"PL — {home_team} -1.5": ev_home_pl,
            f"PL — {away_team} +1.5": ev_away_pl,
            f"Over {total_line}": ev_over,
            f"Under {total_line}": ev_under,
        }
        best_market = max(ev_map, key=lambda k: (ev_map[k] if ev_map[k] is not None else -1e9))
        st.info(f"**Best EV Play:** {best_market}  |  EV {ev_map[best_market]:.2f}%")


# -------------------------
# Player Props Tab
# -------------------------
with tab_player:
    st.subheader("Player Props — Points · Goals · Assists · Shots on Goal")

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player Name", value="Auston Matthews")
        prop_type = st.selectbox(
            "Prop Type",
            [
                "To Record a Point (Yes)",
                "Goals Over",
                "Assists Over",
                "Shots on Goal Over",
            ],
        )
        line = st.number_input("Prop Line (e.g., 0.5, 1.5, 2.5)", value=0.5, step=0.5)
        season_avg = st.number_input("Season Avg (per game for this stat)", value=1.20, step=0.05, min_value=0.0)
        recent_avg = st.number_input("Last 7 Avg (per game)", value=1.40, step=0.05, min_value=0.0)
        weight_recent = st.slider("Recent Weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)

    with col2:
        opp_allowed = st.number_input("Opponent Allowed (per game for this stat)", value=1.10, step=0.05, min_value=0.0)
        league_avg = st.number_input("League Avg for this stat (optional)", value=1.00, step=0.05, min_value=0.0)
        odds_over = st.number_input("Over Odds", value=-120, step=1)
        odds_under = st.number_input("Under Odds", value=-105, step=1)
        sims_player = st.number_input("(Advanced) Simulations for count stats", value=0, step=1000, min_value=0, help="0 = use analytic Poisson/Binomial; set >0 to also Monte Carlo validate.")

    run_player = st.button("Compute Player Prop")

    if run_player:
        base_rate = weighted_rate(season_avg, recent_avg, weight_recent)
        adj_rate = defense_adjust(base_rate, opp_allowed, league_avg if league_avg > 0 else None)

        # True probabilities
        if prop_type == "To Record a Point (Yes)":
            true_over = prob_point_yes(adj_rate)
            true_under = 1.0 - true_over
        else:
            # for Goals/Assists/SOG we use Poisson as default
            true_over = prob_count_over_poisson(adj_rate, line)
            true_under = 1.0 - true_over

        # Implied from odds
        imp_over = american_to_implied(odds_over)
        imp_under = american_to_implied(odds_under)

        # EV
        ev_over = ev_percent(true_over, odds_over)
        ev_under = ev_percent(true_under, odds_under)

        # Output table
        st.markdown("### Results (True %, Implied %, EV %, Tier)")
        rows = [
            {
                "Market": f"{player_name} — {prop_type} {'' if prop_type=='To Record a Point (Yes)' else line}",
                "Side": "Over" if prop_type != "To Record a Point (Yes)" else "Yes",
                "True %": round(true_over * 100, 2),
                "Implied %": round(imp_over * 100, 2) if imp_over is not None else None,
                "EV %": round(ev_over, 2) if ev_over is not None else None,
                "Tier": tier_from_prob(true_over),
            },
            {
                "Market": f"{player_name} — {prop_type} {'' if prop_type=='To Record a Point (Yes)' else line}",
                "Side": "Under" if prop_type != "To Record a Point (Yes)" else "No",
                "True %": round(true_under * 100, 2),
                "Implied %": round(imp_under * 100, 2) if imp_under is not None else None,
                "EV %": round(ev_under, 2) if ev_under is not None else None,
                "Tier": tier_from_prob(true_under),
            },
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Bar charts
        st.markdown("#### Visuals — True vs Implied Probability")
        plot_bar_true_vs_implied(true_over, imp_over, f"{player_name} — Over/Yes")
        plot_bar_true_vs_implied(true_under, imp_under, f"{player_name} — Under/No")

        # Optional simple MC validation for count stats
        if sims_player and prop_type != "To Record a Point (Yes)":
            rng = np.random.default_rng(123)
            samples = rng.poisson(max(adj_rate, 1e-6), size=int(sims_player))
            mc_over = float(np.mean(samples >= math.ceil(line)))
            st.caption(f"Monte Carlo check (n={int(sims_player)}): Over≈ {mc_over*100:.2f}% vs Analytic {true_over*100:.2f}%")

# End of file
