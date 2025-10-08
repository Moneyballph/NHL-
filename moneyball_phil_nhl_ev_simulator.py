# 🏒 Moneyball Phil — NHL EV Simulator (FINAL)
# Tabs: Team Bets · Player Props · Boards/Parlay
# Engines: Poisson (teams) + Poisson/Binomial (player props)
# Visuals: Horizontal progress bars + colored tier badges by TRUE prob
# Inputs: Blank by default. Keys added to prevent duplicate element errors.

import math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------
# Page
# --------------------------
st.set_page_config(page_title="🏒 Moneyball Phil — NHL EV Simulator", layout="wide")
st.title("🏒 Moneyball Phil — NHL EV Simulator")

# --------------------------
# Session State
# --------------------------
def _init_state():
    st.session_state.setdefault("bets_board", [])       # saved single legs
    st.session_state.setdefault("parlay_current", [])   # working parlay legs
    st.session_state.setdefault("parlay_board", [])     # saved parlays
    st.session_state.setdefault("_leg_id", 1)
_init_state()

def _new_leg_id() -> int:
    st.session_state["_leg_id"] += 1
    return st.session_state["_leg_id"]

# --------------------------
# Helpers
# --------------------------
def american_to_implied(odds: float) -> float:
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1.0 + (odds / 100.0) if odds >= 0 else 1.0 + (100.0 / abs(odds))

def decimal_to_american(dec: float) -> float:
    dec = float(dec)
    if dec <= 1.0: return 0.0
    if dec >= 2.0: return (dec - 1.0) * 100.0
    return -100.0 / (dec - 1.0)

def ev_percent(true_prob: float, american_odds: float) -> float:
    if true_prob is None or american_odds is None:
        return None
    p = max(0.0, min(1.0, float(true_prob)))
    o = float(american_odds)
    profit = (o / 100.0) if o >= 0 else (100.0 / abs(o))
    return (p * profit - (1 - p)) * 100.0

def ev_percent_from_decimal(true_prob: float, dec_odds: float) -> float:
    p = max(0.0, min(1.0, float(true_prob)))
    profit = float(dec_odds) - 1.0
    return (p * profit - (1 - p)) * 100.0

# ----- Tier (by TRUE probability, not EV) -----
def tier_from_true_prob(p: float) -> str:
    if p is None: return "—"
    if p > 0.80:  return "Elite"
    if p >= 0.70: return "Strong"
    if p >= 0.60: return "Moderate"
    return "Risky"

def tier_badge_html_from_true(p: float) -> str:
    tier = tier_from_true_prob(p)
    color = {"Elite":"#22c55e", "Strong":"#eab308", "Moderate":"#f97316", "Risky":"#ef4444"}.get(tier, "#9ca3af")
    return f'<span style="background:{color};color:#111;padding:2px 8px;border-radius:8px;font-weight:700">{tier}</span>'

# --------------------------
# Engines (Teams)
# --------------------------
def expected_goals_pair(xgf_home: float, xga_home: float, xgf_away: float, xga_away: float) -> Tuple[float,float]:
    lam_home = (float(xgf_home) + float(xga_away)) / 2.0
    lam_away = (float(xgf_away) + float(xga_home)) / 2.0
    return max(0.05, lam_home), max(0.05, lam_away)

def simulate_matchups(lam_home: float, lam_away: float, n: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    return rng.poisson(lam_home, size=n), rng.poisson(lam_away, size=n)

def ml_pw_over_under_metrics(lam_home: float, lam_away: float, total_line: float, n: int = 10000, seed: int = 42):
    hg, ag = simulate_matchups(lam_home, lam_away, n=n, seed=seed)
    return {
        "home_win": np.mean(hg > ag),
        "away_win": np.mean(ag > hg),
        "home_cover_-1.5": np.mean((hg - ag) >= 2),
        "away_cover_+1.5": np.mean((ag - hg) > -2),
        "over": np.mean((hg + ag) > total_line),
        "under": np.mean((hg + ag) < total_line),
    }

# --------------------------
# Engines (Player Props)
# --------------------------
def weighted_rate(season_avg: float, recent_avg: float, weight_recent: float = 0.7) -> float:
    w = max(0.0, min(1.0, float(weight_recent)))
    return w * float(recent_avg) + (1 - w) * float(season_avg)

def defense_adjust(rate: float, opp_allowed: float, league_avg: float = None) -> float:
    r = float(rate)
    if opp_allowed is None: return max(0.0, r)
    oa = max(0.0, float(opp_allowed))
    if league_avg and league_avg > 0:
        return max(0.0, r * (oa / float(league_avg)))
    return max(0.0, r * (1 + (oa - r) * 0.10 / max(0.1, r)))

def prob_point_yes(rate_points_per_game: float) -> float:
    lam = max(0.0, float(rate_points_per_game))
    return 1.0 - math.exp(-lam)

def prob_count_over_poisson(rate_per_game: float, line: float) -> float:
    k = math.ceil(line)  # >= ceil(line)
    lam = max(rate_per_game, 1e-6)
    s = 0.0
    for i in range(0, k):
        s += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return 1.0 - s

# --------------------------
# Save/Add APIs
# --------------------------
def add_leg_to_board(market: str, side: str, odds: float, true_p: float, implied_p: float, notes: str = ""):
    leg = {
        "id": _new_leg_id(),
        "Market": market,
        "Side": side,
        "Odds (Am)": float(odds),
        "True %": round(100.0 * float(true_p), 2),
        "Implied %": round(100.0 * float(implied_p), 2) if implied_p is not None else None,
        "EV %": round(ev_percent(true_p, odds), 2) if implied_p is not None else None,
        "Tier": tier_from_true_prob(true_p),
        "Notes": notes,
    }
    st.session_state["bets_board"].append(leg)

def add_leg_to_parlay(market: str, side: str, odds: float, true_p: float):
    st.session_state["parlay_current"].append({
        "id": _new_leg_id(),
        "Market": market,
        "Side": side,
        "Odds (Am)": float(odds),
        "True": float(true_p),
        "Decimal": american_to_decimal(float(odds)),
    })

def parlay_metrics(legs: List[Dict]):
    if not legs: return None
    true_comb, dec_prod = 1.0, 1.0
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

# --------------------------
# Layout: Tabs
# --------------------------
tab_team, tab_player = st.tabs(["Team Bets", "Player Props"])

# =====================================================
# Team Bets Tab
# =====================================================
with tab_team:
    st.subheader("Team Bets — Moneyline · Puck Line · Totals")

    colL, colR = st.columns(2)
    with colL:
        home_team = st.text_input("Home Team", value="", key="team_home")
        away_team = st.text_input("Away Team", value="", key="team_away")
        xgf_home = st.number_input("Home xGF (per game)", value=0.0, step=0.1, min_value=0.0, key="xgf_home")
        xga_home = st.number_input("Home xGA (per game)", value=0.0, step=0.1, min_value=0.0, key="xga_home")
        xgf_away = st.number_input("Away xGF (per game)", value=0.0, step=0.1, min_value=0.0, key="xgf_away")
        xga_away = st.number_input("Away xGA (per game)", value=0.0, step=0.1, min_value=0.0, key="xga_away")
        sims = 20000  # hidden default

    with colR:
        st.markdown("**Sportsbook Lines** (enter your current prices)")
        ml_home = st.number_input(f"Moneyline — {home_team or 'Home Team'}", value=0, step=1, key="ml_home")
        ml_away = st.number_input(f"Moneyline — {away_team or 'Away Team'}", value=0, step=1, key="ml_away")

        favorite_team = st.selectbox(
            "Which team is the favorite for the puck line?",
            options=[home_team or "Home", away_team or "Away"],
            index=0, key="favorite_select",
            help="Determines who gets -1.5 (favorite) and who gets +1.5 (dog)."
        )
        if favorite_team == (home_team or "Home"):
            pl_fav_label = f"Puck Line {home_team or 'Home'} -1.5 (odds)"
            pl_dog_label = f"Puck Line {away_team or 'Away'} +1.5 (odds)"
            fav_name, dog_name = (home_team or "Home"), (away_team or "Away")
        else:
            pl_fav_label = f"Puck Line {away_team or 'Away'} -1.5 (odds)"
            pl_dog_label = f"Puck Line {home_team or 'Home'} +1.5 (odds)"
            fav_name, dog_name = (away_team or "Away"), (home_team or "Home")

        pl_fav = st.number_input(pl_fav_label, value=0, step=1, key="pl_fav")
        pl_dog = st.number_input(pl_dog_label, value=0, step=1, key="pl_dog")
        total_line = st.number_input("Total (O/U) line", value=0.0, step=0.5, key="total_line")
        ou_over = st.number_input("Over Odds", value=0, step=1, key="ou_over")
        ou_under = st.number_input("Under Odds", value=0, step=1, key="ou_under")

    run_team = st.button("🔮 Run Team Projection", key="run_team")
    if run_team:
        # --- projections
        lam_h, lam_a = expected_goals_pair(xgf_home, xga_home, xgf_away, xga_away)
        metrics = ml_pw_over_under_metrics(lam_h, lam_a, total_line, n=int(sims))

        # True probs
        p_home = metrics["home_win"]
        p_away = metrics["away_win"]
        p_over, p_under = metrics["over"], metrics["under"]
        # puck line mapped to labels
        # puck line mapped to labels (favorite gets the -1.5 side)
        if fav_name == (home_team or "Home"):
           p_fav_pl = metrics["home_cover_-1.5"]
           p_dog_pl = metrics["away_cover_+1.5"]
        else:
           p_fav_pl = metrics["away_cover_-1.5"]
           p_dog_pl = metrics["home_cover_+1.5"]


        # Implied
        imp_home, imp_away = american_to_implied(ml_home), american_to_implied(ml_away)
        imp_favpl, imp_dogpl = american_to_implied(pl_fav), american_to_implied(pl_dog)
        imp_over, imp_under = american_to_implied(ou_over), american_to_implied(ou_under)

        # EVs (we still show, but tiers are by TRUE)
        ev_home, ev_away = ev_percent(p_home, ml_home), ev_percent(p_away, ml_away)
        ev_favpl, ev_dogpl = ev_percent(p_fav_pl, pl_fav), ev_percent(p_dog_pl, pl_dog)
        ev_overv, ev_underv = ev_percent(p_over, ou_over), ev_percent(p_under, ou_under)

        # --- projected goals/total/margin (shown like ATS module)
        proj_home, proj_away = lam_h, lam_a
        proj_total, proj_margin = proj_home + proj_away, proj_home - proj_away
        st.subheader("Projected Game Outcome")
        st.markdown(
            f"**Projected {home_team or 'Home'}:** {proj_home:.2f}  |  "
            f"**Projected {away_team or 'Away'}:** {proj_away:.2f}  |  "
            f"**Projected Total:** {proj_total:.2f}  |  "
            f"**Projected Margin:** {proj_margin:.2f}"
        )

        # --- inline summaries (ranked by TRUE%)
        inline = [
            (f"ML — {home_team or 'Home'}", p_home, imp_home, ev_home, ("ML", ml_home)),
            (f"ML — {away_team or 'Away'}", p_away, imp_away, ev_away, ("ML", ml_away)),
            (pl_fav_label.replace(" (odds)", ""), p_fav_pl, imp_favpl, ev_favpl, ("PL", pl_fav)),
            (pl_dog_label.replace(" (odds)", ""), p_dog_pl, imp_dogpl, ev_dogpl, ("PL", pl_dog)),
            (f"Over {total_line}", p_over, imp_over, ev_overv, ("Over", ou_over)),
            (f"Under {total_line}", p_under, imp_under, ev_underv, ("Under", ou_under)),
        ]
        inline.sort(key=lambda t: (t[1] if t[1] is not None else 0.0), reverse=True)

        st.subheader("Inline Results (ranked by True %)")
        for label, tp, ip, evv, (side_tag, odds_val) in inline:
            badge = tier_badge_html_from_true(tp)
            ip_pct = (ip * 100.0) if ip is not None else None
            st.markdown(
                f"🔹 **{label}** → True **{tp*100:.2f}%** | "
                f"Implied **{ip_pct:.2f}%** | EV **{(evv if evv is not None else 0.0):.2f}%** {badge}",
                unsafe_allow_html=True
            )
            st.progress(tp, text=f"True: {tp*100:.2f}%")
            if ip is not None:
                st.progress(ip, text=f"Implied: {ip*100:.2f}%")

        # --- table with colored Tier column (HTML)
        def _row(market, tp, ip, evv):
            return {
                "Market": market,
                "True %": round(tp * 100, 2),
                "Implied %": round(ip * 100, 2) if ip is not None else None,
                "EV %": round(evv, 2) if evv is not None else None,
                "Tier": tier_badge_html_from_true(tp)
            }

        table_rows = [
            _row(f"ML — {home_team or 'Home'}", p_home, imp_home, ev_home),
            _row(f"ML — {away_team or 'Away'}", p_away, imp_away, ev_away),
            _row(pl_fav_label.replace(" (odds)", ""), p_fav_pl, imp_favpl, ev_favpl),
            _row(pl_dog_label.replace(" (odds)", ""), p_dog_pl, imp_dogpl, ev_dogpl),
            _row(f"Over {total_line}", p_over, imp_over, ev_overv),
            _row(f"Under {total_line}", p_under, imp_under, ev_underv),
        ]
        df_team = pd.DataFrame(table_rows).sort_values("True %", ascending=False).reset_index(drop=True)
        st.subheader("Bet Results (sorted by True %)")
        st.write(df_team.to_html(escape=False, index=False), unsafe_allow_html=True)

        # --- Add to Board / Parlay quick actions
        st.markdown("**Quick Actions**")
        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            if st.button("Add ML Home to Board", key="add_ml_home_board"):
                add_leg_to_board(f"ML — {home_team or 'Home'}", "ML", ml_home, p_home, imp_home)
            if st.button("Add PL Favorite to Board", key="add_pl_fav_board"):
                add_leg_to_board(pl_fav_label.replace(" (odds)", ""), "PL", pl_fav, p_fav_pl, imp_favpl)
            if st.button("Add Over to Board", key="add_over_board"):
                add_leg_to_board(f"Over {total_line}", "Over", ou_over, p_over, imp_over)
        with ca2:
            if st.button("Add ML Away to Board", key="add_ml_away_board"):
                add_leg_to_board(f"ML — {away_team or 'Away'}", "ML", ml_away, p_away, imp_away)
            if st.button("Add PL Dog to Board", key="add_pl_dog_board"):
                add_leg_to_board(pl_dog_label.replace(" (odds)", ""), "PL", pl_dog, p_dog_pl, imp_dogpl)
            if st.button("Add Under to Board", key="add_under_board"):
                add_leg_to_board(f"Under {total_line}", "Under", ou_under, p_under, imp_under)
        with ca3:
            st.caption("Add legs to current parlay:")
            if st.button("+ ML Home", key="add_ml_home_parlay"):
                add_leg_to_parlay(f"ML — {home_team or 'Home'}", "ML", ml_home, p_home)
            if st.button("+ ML Away", key="add_ml_away_parlay"):
                add_leg_to_parlay(f"ML — {away_team or 'Away'}", "ML", ml_away, p_away)
            if st.button("+ PL Favorite -1.5", key="add_pl_fav_parlay"):
                add_leg_to_parlay(pl_fav_label.replace(" (odds)", ""), "PL", pl_fav, p_fav_pl)
            if st.button("+ PL Dog +1.5", key="add_pl_dog_parlay"):
                add_leg_to_parlay(pl_dog_label.replace(" (odds)", ""), "PL", pl_dog, p_dog_pl)
            if st.button("+ Over", key="add_over_parlay"):
                add_leg_to_parlay(f"Over {total_line}", "Over", ou_over, p_over)
            if st.button("+ Under", key="add_under_parlay"):
                add_leg_to_parlay(f"Under {total_line}", "Under", ou_under, p_under)

# =====================================================
# Player Props Tab
# =====================================================
with tab_player:
    st.subheader("Player Props — Points · Goals · Assists · Shots on Goal")

    col1, col2 = st.columns(2)
    with col1:
        player_name = st.text_input("Player Name", value="", key="pp_player")
        prop_type = st.selectbox(
            "Prop Type",
            ["To Record a Point (Yes)", "Goals Over", "Assists Over", "Shots on Goal Over"],
            key="pp_type"
        )
        line = st.number_input("Prop Line (e.g., 0.5, 1.5, 2.5)", value=0.0, step=0.5, key="pp_line")
        season_avg = st.number_input("Season Avg (per game for this stat)", value=0.0, step=0.05, min_value=0.0, key="pp_season")
        recent_avg = st.number_input("Last 7 Avg (per game)", value=0.0, step=0.05, min_value=0.0, key="pp_recent")
        weight_recent = st.slider("Recent Weight", 0.0, 1.0, 0.70, 0.05, key="pp_weight")

    with col2:
        opp_allowed = st.number_input("Opponent Allowed (per game for this stat)", value=0.0, step=0.05, min_value=0.0, key="pp_opp")
        league_avg = st.number_input("League Avg for this stat (optional)", value=0.0, step=0.05, min_value=0.0, key="pp_lg")
        odds_over = st.number_input("Over Odds", value=0, step=1, key="pp_over_odds")
        odds_under = st.number_input("Under Odds", value=0, step=1, key="pp_under_odds")

    run_player = st.button("🔮 Compute Player Prop", key="pp_run")
    if run_player:
        base_rate = weighted_rate(season_avg, recent_avg, weight_recent)
        adj_rate = defense_adjust(base_rate, opp_allowed, league_avg if league_avg > 0 else None)

        if prop_type == "To Record a Point (Yes)":
            # If line is 0.5, use P(≥1); otherwise use Poisson for ≥ ceil(line)
            if float(line) <= 0.51:
                true_over = prob_point_yes(adj_rate)
            else:
                true_over = prob_count_over_poisson(adj_rate, line)
        else:
            # Goals / Assists / SOG already use the count model
            true_over = prob_count_over_poisson(adj_rate, line)

        true_under = 1.0 - true_over



        imp_over, imp_under = american_to_implied(odds_over), american_to_implied(odds_under)
        ev_over, ev_under = ev_percent(true_over, odds_over), ev_percent(true_under, odds_under)

        # Projection headline (like ATS module)
        st.subheader("Projected Player Rate")
        st.markdown(
            f"**Weighted Rate (season/recent):** {base_rate:.3f}  |  "
            f"**Defense-adjusted Rate:** {adj_rate:.3f}  |  "
            f"**Prop Line:** {line:.1f}"
        )

        # Inline (ranked by TRUE)
        inline_pp = [
            (f"{player_name} — {prop_type}", "Over/Yes", true_over, imp_over, ev_over, odds_over),
            (f"{player_name} — {prop_type}", "Under/No", true_under, imp_under, ev_under, odds_under),
        ]
        inline_pp.sort(key=lambda t: (t[2] if t[2] is not None else 0.0), reverse=True)

        st.subheader("Inline Results (ranked by True %)")
        for label, side, tp, ip, evv, odds_val in inline_pp:
            badge = tier_badge_html_from_true(tp)
            ip_pct = (ip * 100.0) if ip is not None else None
            st.markdown(
                f"🔹 **{label} — {side}** → True **{tp*100:.2f}%** | "
                f"Implied **{ip_pct:.2f}%** | EV **{(evv if evv is not None else 0.0):.2f}%** {badge}",
                unsafe_allow_html=True
            )
            st.progress(tp, text=f"True: {tp*100:.2f}%")
            if ip is not None:
                st.progress(ip, text=f"Implied: {ip*100:.2f}%")

        # Results table (with colored Tier)
        df_pp = pd.DataFrame([
            {
                "Market": f"{player_name} — {prop_type}",
                "Side": "Over/Yes",
                "True %": round(true_over * 100, 2),
                "Implied %": round(imp_over * 100, 2) if imp_over is not None else None,
                "EV %": round(ev_over, 2) if ev_over is not None else None,
                "Tier": tier_badge_html_from_true(true_over)
            },
            {
                "Market": f"{player_name} — {prop_type}",
                "Side": "Under/No",
                "True %": round(true_under * 100, 2),
                "Implied %": round(imp_under * 100, 2) if imp_under is not None else None,
                "EV %": round(ev_under, 2) if ev_under is not None else None,
                "Tier": tier_badge_html_from_true(true_under)
            },
        ]).sort_values("True %", ascending=False).reset_index(drop=True)
        st.subheader("Bet Results (sorted by True %)")
        st.write(df_pp.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Quick actions
        q1, q2 = st.columns(2)
        with q1:
            if st.button("Add Over/Yes to Board", key="pp_add_over_board"):
                add_leg_to_board(f"{player_name} — {prop_type}", "Over/Yes", odds_over, true_over, imp_over)
            if st.button("+ Add Over/Yes to Parlay", key="pp_add_over_parlay"):
                add_leg_to_parlay(f"{player_name} — {prop_type}", "Over/Yes", odds_over, true_over)
        with q2:
            if st.button("Add Under/No to Board", key="pp_add_under_board"):
                add_leg_to_board(f"{player_name} — {prop_type}", "Under/No", odds_under, true_under, imp_under)
            if st.button("+ Add Under/No to Parlay", key="pp_add_under_parlay"):
                add_leg_to_parlay(f"{player_name} — {prop_type}", "Under/No", odds_under, true_under)

# =====================================================
# Boards & Parlay Builder
# =====================================================
st.markdown("---")
st.header("📋 Bet Boards & Parlay Builder")

# Single bets board
st.subheader("Saved Single Bets")
if st.session_state["bets_board"]:
    st.dataframe(pd.DataFrame(st.session_state["bets_board"]), use_container_width=True)
    if st.button("Clear Single-Bet Board", key="clear_single_board"):
        st.session_state["bets_board"] = []
else:
    st.caption("No single bets saved yet.")

# Current parlay area
st.subheader("Current Parlay (Working)")
if st.session_state["parlay_current"]:
    df_parlay = pd.DataFrame(st.session_state["parlay_current"])
    st.dataframe(df_parlay, use_container_width=True)
    met = parlay_metrics(st.session_state["parlay_current"])
    if met:
        st.write(
            f"**Combined True %:** {met['True %']:.2f}%  |  "
            f"**Implied %:** {met['Implied %']:.2f}%  |  "
            f"**Odds (Am):** {met['Odds (Am)']:.0f}  |  "
            f"**EV:** {met['EV %']:.2f}%"
        )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save Parlay to Board", key="save_parlay"):
            st.session_state["parlay_board"].append({
                "Legs": [l["Market"] for l in st.session_state["parlay_current"]],
                "Odds (Am)": met["Odds (Am)"] if met else None,
                "Decimal": met["Decimal"] if met else None,
                "True %": met["True %"] if met else None,
                "Implied %": met["Implied %"] if met else None,
                "EV %": met["EV %"] if met else None,
            })
            st.session_state["parlay_current"] = []
    with c2:
        if st.button("Clear Current Parlay", key="clear_parlay"):
            st.session_state["parlay_current"] = []
else:
    st.caption("No legs in the current parlay yet. Add legs from Team Bets or Player Props above.")

# Saved parlays
st.subheader("Saved Parlays Board")
if st.session_state["parlay_board"]:
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
    if st.button("Clear Saved Parlays", key="clear_saved_parlays"):
        st.session_state["parlay_board"] = []
else:
    st.caption("No parlays saved yet.")

