# 🏒 Moneyball Phil — NHL EV Simulator (Standalone)
# Tabs: Team Bets (Moneyline / Puck Line / Totals), Player Props
# Engines: Poisson (teams) + Poisson/Binomial (player props)
# Extras: Bar charts, Add-to-Board, Parlay Builder (add legs, combined true %, implied %, EV %, save parlay)
# Dependencies: streamlit, numpy, pandas, matplotlib

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
    st.session_state.setdefault("bets_board", [])  # list of single-leg bets saved
    st.session_state.setdefault("parlay_current", [])  # working parlay legs
    st.session_state.setdefault("parlay_board", [])  # saved parlays
    st.session_state.setdefault("_leg_id", 1)

_init_state()

# =========================
# ===== Helper Utils ======
# =========================

def american_to_implied(odds: float) -> float:
    if odds is None:
        return None
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds >= 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def decimal_to_american(dec: float) -> float:
    dec = float(dec)
    if dec <= 1.0:
        return 0.0
    if dec >= 2.0:
        return (dec - 1.0) * 100.0
    # negative odds
    return -100.0 / (dec - 1.0)


def ev_percent(true_prob: float, american_odds: float) -> float:
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

def expected_goals_pair(xgf_home: float, xga_home: float, xgf_away: float, xga_away: float) -> Tuple[float, float]:
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
    draw_reg = np.mean(hg == ag)
    home_cover_m15 = np.mean((hg - ag) >= 2)
    away_cover_p15 = np.mean((ag - hg) > -2)
    totals_over = np.mean((hg + ag) > total_line)
    totals_under = np.mean((hg + ag) < total_line)
    totals_push = 1.0 - totals_over - totals_under
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
    r = float(rate)
    if opp_allowed is None:
        return max(0.0, r)
    oa = max(0.0, float(opp_allowed))
    if league_avg is None or league_avg <= 0:
        factor = 1.0 + (oa - r) * 0.10 / max(0.1, r)
        return max(0.0, r * factor)
    factor = oa / float(league_avg)
    return max(0.0, r * factor)


def prob_point_yes(rate_points_per_game: float) -> float:
    lam = max(0.0, float(rate_points_per_game))
    return 1.0 - math.exp(-lam)


def prob_count_over_poisson(rate_per_game: float, line: float) -> float:
    k = math.floor(line + 1e-9)
    if abs(line - (k + 0.5)) < 0.49:
        k = math.ceil(line)
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
# ====== Add/Save APIs =====
# =========================

def _new_leg_id() -> int:
    st.session_state["_leg_id"] += 1
    return st.session_state["_leg_id"]


def add_leg_to_board(market: str, side: str, odds: float, true_p: float, implied_p: float, notes: str = ""):
    leg = {
        "id": _new_leg_id(),
        "Market": market,
        "Side": side,
        "Odds (Am)": float(odds),
        "True %": round(100.0 * float(true_p), 2),
        "Implied %": round(100.0 * float(implied_p), 2) if implied_p is not None else None,
        "EV %": round(ev_percent(true_p, odds), 2) if implied_p is not None else None,
        "Tier": tier_from_prob(true_p),
        "Notes": notes,
    }
    st.session_state["bets_board"].append(leg)


def add_leg_to_parlay(market: str, side: str, odds: float, true_p: float):
    leg = {
        "id": _new_leg_id(),
        "Market": market,
        "Side": side,
        "Odds (Am)": float(odds),
        "True": float(true_p),
        "Decimal": american_to_decimal(float(odds)),
    }
    st.session_state["parlay_current"].append(leg)


def parlay_metrics(legs: List[Dict]):
    if not legs:
        return None
    # Combined true probability (independence assumption)
    true_comb = 1.0
    dec_prod = 1.0
    for lg in legs:
        true_comb *= max(0.0, min(1.0, float(lg["True"])))
        dec_prod *= float(lg["Decimal"])
    implied_comb = 1.0 / dec_prod  # from decimal odds
    am_comb = decimal_to_american(dec_prod)
    ev_comb = ev_percent_from_decimal(true_comb, dec_prod)
    return {
        "True %": true_comb * 100.0,
        "Implied %": implied_comb * 100.0,
        "Decimal": dec_prod,
        "Odds (Am)": am_comb,
        "EV %": ev_comb,
    }

# =========================
# ========= Layout =========
# =========================

tab_team, tab_player = st.tabs(["Team Bets", "Player Props"])

# -------------------------
# Team Bets Tab
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
        sims = 20000  # hidden default simulation count for Poisson model


   with colR:
    st.markdown("**Sportsbook Lines** (enter your current prices)")

    ml_home = st.number_input(f"Moneyline — {home_team}", value=-135, step=1)
    ml_away = st.number_input(f"Moneyline — {away_team}", value=+115, step=1)

    # 🏒 Dynamic Puck Line favorite selection
    favorite_team = st.selectbox(
        "Which team is the favorite for the puck line?",
        options=[home_team, away_team],
        index=0,
        help="This determines which side gets -1.5 and which gets +1.5."
    )

    if favorite_team == home_team:
        pl_fav_label = f"Puck Line {home_team} -1.5 (odds)"
        pl_dog_label = f"Puck Line {away_team} +1.5 (odds)"
    else:
        pl_fav_label = f"Puck Line {away_team} -1.5 (odds)"
        pl_dog_label = f"Puck Line {home_team} +1.5 (odds)"

    pl_fav_odds = st.number_input(pl_fav_label, value=+150, step=1)
    pl_dog_odds = st.number_input(pl_dog_label, value=-170, step=1)

    total_line = st.number_input("Total (O/U) line", value=6.5, step=0.5)
    ou_over = st.number_input("Over Odds", value=+105, step=1)
    ou_under = st.number_input("Under Odds", value=-125, step=1)



    run_team = st.button("Run Team Simulation")

    if run_team:
        lam_h, lam_a = expected_goals_pair(xgf_home, xga_home, xgf_away, xga_away)
        metrics = ml_pw_over_under_metrics(lam_h, lam_a, total_line, n=int(sims))

        p_home = metrics["home_win"]
        p_away = metrics["away_win"]
        p_home_pl = metrics["home_cover_-1.5"]
        p_away_pl = metrics["away_cover_+1.5"]
        p_over = metrics["over"]
        p_under = metrics["under"]

        imp_home = american_to_implied(ml_home)
        imp_away = american_to_implied(ml_away)
        imp_home_pl = american_to_implied(pl_home)
        imp_away_pl = american_to_implied(pl_away)
        imp_over = american_to_implied(ou_over)
        imp_under = american_to_implied(ou_under)

        ev_home = ev_percent(p_home, ml_home)
        ev_away = ev_percent(p_away, ml_away)
        ev_home_pl = ev_percent(p_home_pl, pl_home)
        ev_away_pl = ev_percent(p_away_pl, pl_away)
        ev_over = ev_percent(p_over, ou_over)
        ev_under = ev_percent(p_under, ou_under)

        st.markdown("### Results (True %, Implied %, EV %, Tier)")
        team_rows = [
            {"Market": f"ML — {home_team or 'Home'}", "True %": round(p_home*100,2), "Implied %": round(imp_home*100,2) if imp_home is not None else None, "EV %": round(ev_home,2) if ev_home is not None else None, "Tier": tier_from_prob(p_home)},
            {"Market": f"ML — {away_team or 'Away'}", "True %": round(p_away*100,2), "Implied %": round(imp_away*100,2) if imp_away is not None else None, "EV %": round(ev_away,2) if ev_away is not None else None, "Tier": tier_from_prob(p_away)},
            {"Market": f"Puck Line — {home_team or 'Home'} -1.5", "True %": round(p_home_pl*100,2), "Implied %": round(imp_home_pl*100,2) if imp_home_pl is not None else None, "EV %": round(ev_home_pl,2) if ev_home_pl is not None else None, "Tier": tier_from_prob(p_home_pl)},
            {"Market": f"Puck Line — {away_team or 'Away'} +1.5", "True %": round(p_away_pl*100,2), "Implied %": round(imp_away_pl*100,2) if imp_away_pl is not None else None, "EV %": round(ev_away_pl,2) if ev_away_pl is not None else None, "Tier": tier_from_prob(p_away_pl)},
            {"Market": f"Total Over {total_line}", "True %": round(p_over*100,2), "Implied %": round(imp_over*100,2) if imp_over is not None else None, "EV %": round(ev_over,2) if ev_over is not None else None, "Tier": tier_from_prob(p_over)},
            {"Market": f"Total Under {total_line}", "True %": round(p_under*100,2), "Implied %": round(imp_under*100,2) if imp_under is not None else None, "EV %": round(ev_under,2) if ev_under is not None else None, "Tier": tier_from_prob(p_under)},
        ]
        df_team = pd.DataFrame(team_rows)
        st.dataframe(df_team, use_container_width=True)

        # Action buttons per market
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Add ML (Home) to Board"):
                add_leg_to_board(f"ML — {home_team or 'Home'}", "ML", ml_home, p_home, imp_home)
            if st.button("Add PL Home -1.5 to Board"):
                add_leg_to_board(f"PL — {home_team or 'Home'} -1.5", "PL -1.5", pl_home, p_home_pl, imp_home_pl)
            if st.button("Add Over to Board"):
                add_leg_to_board(f"Over {total_line}", "Over", ou_over, p_over, imp_over)
        with c2:
            if st.button("Add ML (Away) to Board"):
                add_leg_to_board(f"ML — {away_team or 'Away'}", "ML", ml_away, p_away, imp_away)
            if st.button("Add PL Away +1.5 to Board"):
                add_leg_to_board(f"PL — {away_team or 'Away'} +1.5", "PL +1.5", pl_away, p_away_pl, imp_away_pl)
            if st.button("Add Under to Board"):
                add_leg_to_board(f"Under {total_line}", "Under", ou_under, p_under, imp_under)
        with c3:
            st.markdown("**Add to Parlay**")
            if st.button("+ ML Home"):
                add_leg_to_parlay(f"ML — {home_team or 'Home'}", "ML", ml_home, p_home)
            if st.button("+ ML Away"):
                add_leg_to_parlay(f"ML — {away_team or 'Away'}", "ML", ml_away, p_away)
            if st.button("+ PL Home -1.5"):
                add_leg_to_parlay(f"PL — {home_team or 'Home'} -1.5", "PL -1.5", pl_home, p_home_pl)
            if st.button("+ PL Away +1.5"):
                add_leg_to_parlay(f"PL — {away_team or 'Away'} +1.5", "PL +1.5", pl_away, p_away_pl)
            if st.button("+ Over"):
                add_leg_to_parlay(f"Over {total_line}", "Over", ou_over, p_over)
            if st.button("+ Under"):
                add_leg_to_parlay(f"Under {total_line}", "Under", ou_under, p_under)

        st.markdown("#### Visuals — True vs Implied Probability")
        colA, colB, colC = st.columns(3)
        with colA:
            plot_bar_true_vs_implied(p_home, imp_home, f"ML — {home_team or 'Home'}")
            plot_bar_true_vs_implied(p_away, imp_away, f"ML — {away_team or 'Away'}")
        with colB:
            plot_bar_true_vs_implied(p_home_pl, imp_home_pl, f"PL {home_team or 'Home'} -1.5")
            plot_bar_true_vs_implied(p_away_pl, imp_away_pl, f"PL {away_team or 'Away'} +1.5")
        with colC:
            plot_bar_true_vs_implied(p_over, imp_over, f"Over {total_line}")
            plot_bar_true_vs_implied(p_under, imp_under, f"Under {total_line}")

        ev_map = {
            f"ML — {home_team or 'Home'}": ev_home,
            f"ML — {away_team or 'Away'}": ev_away,
            f"PL — {home_team or 'Home'} -1.5": ev_home_pl,
            f"PL — {away_team or 'Away'} +1.5": ev_away_pl,
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
        player_name = st.text_input("Player Name", value="")
        prop_type = st.selectbox("Prop Type", ["To Record a Point (Yes)", "Goals Over", "Assists Over", "Shots on Goal Over"])
        line = st.number_input("Prop Line (e.g., 0.5, 1.5, 2.5)", value=0.0, step=0.5)
        season_avg = st.number_input("Season Avg (per game for this stat)", value=0.0, step=0.05, min_value=0.0)
        recent_avg = st.number_input("Last 7 Avg (per game)", value=0.0, step=0.05, min_value=0.0)
        weight_recent = st.slider("Recent Weight", min_value=0.0, max_value=1.0, value=0.70, step=0.05)

    with col2:
        opp_allowed = st.number_input("Opponent Allowed (per game for this stat)", value=0.0, step=0.05, min_value=0.0)
        league_avg = st.number_input("League Avg for this stat (optional)", value=0.0, step=0.05, min_value=0.0)
        odds_over = st.number_input("Over Odds", value=0, step=1, key="player_over_odds")
        odds_under = st.number_input("Under Odds", value=0, step=1, key="player_under_odds")
        sims_player = st.number_input("(Advanced) Simulations for count stats", value=0, step=1000, min_value=0)


    run_player = st.button("Compute Player Prop")

    if run_player:
        base_rate = weighted_rate(season_avg, recent_avg, weight_recent)
        adj_rate = defense_adjust(base_rate, opp_allowed, league_avg if league_avg > 0 else None)

        if prop_type == "To Record a Point (Yes)":
            true_over = prob_point_yes(adj_rate)
            true_under = 1.0 - true_over
        else:
            true_over = prob_count_over_poisson(adj_rate, line)
            true_under = 1.0 - true_over

        imp_over = american_to_implied(odds_over)
        imp_under = american_to_implied(odds_under)

        ev_over = ev_percent(true_over, odds_over)
        ev_under = ev_percent(true_under, odds_under)

        st.markdown("### Results (True %, Implied %, EV %, Tier)")
        rows = [
            {"Market": f"{player_name} — {prop_type} {'' if prop_type=='To Record a Point (Yes)' else line}", "Side": "Over" if prop_type != "To Record a Point (Yes)" else "Yes", "True %": round(true_over*100,2), "Implied %": round(imp_over*100,2) if imp_over is not None else None, "EV %": round(ev_over,2) if ev_over is not None else None, "Tier": tier_from_prob(true_over)},
            {"Market": f"{player_name} — {prop_type} {'' if prop_type=='To Record a Point (Yes)' else line}", "Side": "Under" if prop_type != "To Record a Point (Yes)" else "No", "True %": round(true_under*100,2), "Implied %": round(imp_under*100,2) if imp_under is not None else None, "EV %": round(ev_under,2) if ev_under is not None else None, "Tier": tier_from_prob(true_under)},
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        cA, cB, cC = st.columns(3)
        with cA:
            if st.button("Add Over/Yes to Board"):
                add_leg_to_board(rows[0]["Market"], rows[0]["Side"], odds_over, true_over, imp_over)
            if st.button("+ Add Over/Yes to Parlay"):
                add_leg_to_parlay(rows[0]["Market"], rows[0]["Side"], odds_over, true_over)
        with cB:
            if st.button("Add Under/No to Board"):
                add_leg_to_board(rows[1]["Market"], rows[1]["Side"], odds_under, true_under, imp_under)
            if st.button("+ Add Under/No to Parlay"):
                add_leg_to_parlay(rows[1]["Market"], rows[1]["Side"], odds_under, true_under)

        st.markdown("#### Visuals — True vs Implied Probability")
        plot_bar_true_vs_implied(true_over, imp_over, f"{player_name} — Over/Yes")
        plot_bar_true_vs_implied(true_under, imp_under, f"{player_name} — Under/No")

        if sims_player and prop_type != "To Record a Point (Yes)":
            rng = np.random.default_rng(123)
            samples = rng.poisson(max(adj_rate, 1e-6), size=int(sims_player))
            mc_over = float(np.mean(samples >= math.ceil(line)))
            st.caption(f"Monte Carlo check (n={int(sims_player)}): Over≈ {mc_over*100:.2f}% vs Analytic {true_over*100:.2f}%")

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
    df_parlay = pd.DataFrame(st.session_state["parlay_current"])  # shows legs
    st.dataframe(df_parlay, use_container_width=True)
    met = parlay_metrics(st.session_state["parlay_current"])
    if met:
        st.write(f"**Combined True %:** {met['True %']:.2f}%  |  **Implied %:** {met['Implied %']:.2f}%  |  **Odds (Am):** {met['Odds (Am)']:.0f}  |  **EV:** {met['EV %']:.2f}%")
    colx, coly, colz = st.columns([1,1,1])
    with colx:
        if st.button("Save Parlay to Board"):
            st.session_state["parlay_board"].append({
                "Legs": [l["Market"] for l in st.session_state["parlay_current"]],
                "Odds (Am)": met["Odds (Am)"] if met else None,
                "Decimal": met["Decimal"] if met else None,
                "True %": met["True %"] if met else None,
                "Implied %": met["Implied %"] if met else None,
                "EV %": met["EV %"] if met else None,
            })
            st.session_state["parlay_current"] = []
    with coly:
        if st.button("Clear Current Parlay"):
            st.session_state["parlay_current"] = []
else:
    st.caption("No legs in the current parlay yet. Add legs from Team Bets or Player Props above.")

# Saved Parlays Board
st.subheader("Saved Parlays Board")
if st.session_state["parlay_board"]:
    # expand legs into readable strings
    formatted = []
    for i, p in enumerate(st.session_state["parlay_board"], start=1):
        formatted.append({
            "#": i,
            "Legs": " | ".join(p["Legs"]),
            "Odds (Am)": round(p["Odds (Am)"], 0) if p["Odds (Am)"] is not None else None,
            "True %": round(p["True %"], 2) if p["True %"] is not None else None,
            "Implied %": round(p["Implied %"], 2) if p["Implied %"] is not None else None,
            "EV %": round(p["EV %"], 2) if p["EV %"] is not None else None,
        })
    st.dataframe(pd.DataFrame(formatted), use_container_width=True)
    if st.button("Clear Saved Parlays"):
        st.session_state["parlay_board"] = []
else:
    st.caption("No parlays saved yet.")

# End of file
