"""
Phase 5: Claims Review Desk, the adjuster's web app.

    streamlit run app.py

Four tabs:
  Review queue    claims waiting for a human: photo, AI recommendation, policy quotes,
                  red flags, and the adjuster's decision (signed, with a note)
  Process a claim run a new claim through the agents and watch each step live
  Audit trail     every step of a claim, who decided, and any override
  Results         the held-out test results (test 1 vs tests 2 + 3)

All logic lives in adjuster_service.py; this file is only the screen.
"""

import os

import streamlit as st

import adjuster_service as svc

st.set_page_config(page_title="Claims Review Desk", page_icon="🧾", layout="wide")

REC_COLOR = {"approve": "green", "investigate": "orange", "deny": "red"}
SEV_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
STATUS_TEXT = {"not_started": "not processed", "waiting_adjuster": "waiting for adjuster",
               "waiting_documents": "waiting for documents", "decided": "decided",
               "closed": "closed (incomplete)", "in_progress": "in progress"}


@st.cache_resource
def get_graph():
    return svc.open_graph()


def pkr(x):
    return f"PKR {x:,.0f}" if isinstance(x, (int, float)) else "—"


def rec_badge(rec, label=None):
    if rec:
        st.badge((label or rec).upper(), color=REC_COLOR.get(rec, "gray"))


graph = get_graph()

# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧾 Claims Review Desk")
    st.caption("Motor claims · multi-agent review · a human makes every final decision")
    reviewer = st.text_input("Your name (signs each decision)", value=st.session_state.get("reviewer", "adjuster"))
    st.session_state["reviewer"] = reviewer
    use_llm = st.toggle("Use AI agents", value=True,
                        help="On: the AI reads the policy wording, checks the claimant's story and writes the "
                             "note (answers for dev claims are cached, so usually free). "
                             "Off: rules only, always free.")
    st.divider()
    st.caption("System settings (from .env)")
    st.code("\n".join(f"{k}={os.getenv(k, '(default)')}" for k in
                      ("AI_CHECKS", "PHOTO_PROMPT", "AI_ESCALATION", "TYRE_REVIEW", "OPENAI_MODEL")),
            language=None)

tab_queue, tab_process, tab_audit, tab_results = st.tabs(
    ["📥 Review queue", "▶️ Process a claim", "📜 Audit trail", "📊 Results"])

rows = svc.list_claims(graph)
by_status = {}
for r in rows:
    by_status.setdefault(r["status"], []).append(r)


def claim_label(r):
    parts = [r["claim"]]
    if r["claimant"]:
        parts.append(r["claimant"])
    if r["amount_pkr"]:
        parts.append(pkr(r["amount_pkr"]))
    if r["recommendation"] and r["status"] == "waiting_adjuster":
        parts.append(f"AI: {r['recommendation']}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# review queue
# ---------------------------------------------------------------------------
def show_review(cid):
    state = svc.claim_state(graph, cid)
    d = svc.details(state["values"])
    form, w, cov, ev = d["form"], d["writeup"], d["coverage"], d["evidence"]

    left, right = st.columns([2, 3], gap="large")
    with left:
        if d["photo_path"] and os.path.exists(d["photo_path"]):
            st.image(d["photo_path"], caption="Damage photo submitted with the claim", width="stretch")
        p = d["photo"]
        st.markdown(f"**Photo reading:** {p.get('severity', '?')} {p.get('damage_type', '?')} "
                    f"on {p.get('damaged_part') or '?'}")
        if len(p.get("all_damages") or []) > 1:
            st.caption("; ".join(f"{x['severity']} {x['damage_type']} on {x['part']}" for x in p["all_damages"]))
        with st.container(border=True):
            st.markdown(f"**{form.get('claimant_name', '')}** · policy `{form.get('policy_number', '')}`")
            st.markdown(f"{form.get('vehicle', '')} · incident {form.get('incident_date', '')} "
                        f"at {form.get('location', '')}")
            st.markdown(f"**Amount claimed:** {pkr(form.get('amount_claimed'))}")
            st.caption(f"Claimant's description (untrusted): {form.get('description', '')}")

    with right:
        top = st.columns([1, 3])
        with top[0]:
            st.caption("AI recommendation")
            rec_badge(d["recommendation"])
        with top[1]:
            st.markdown(f"**{w.get('summary', '')}**")
        st.write(w.get("justification", ""))

        if d["flags"]:
            st.markdown("**Red flags**")
            for f in d["flags"]:
                st.markdown(f"{SEV_ICON.get(f['severity'], '•')} `{f['code']}` ({f['severity']}, {f['from']}) "
                            f"— {f['message']}")
        else:
            st.markdown("**Red flags:** none")

        with st.expander(f"Coverage: {cov.get('status', '?')} ({cov.get('method', '')})", expanded=True):
            for reason in cov.get("reasons", []):
                st.markdown(f"- {reason}")
            for q in d["quotes"]:
                st.markdown(f"> {q['quote']}")
                st.caption(f"✔ verified word-for-word in {q['policy_file']}, page {q['page']} · {q['section']}")
        with st.expander(f"Evidence: repair cost {ev.get('cost_status', '?')}"):
            rng = ev.get("typical_range_pkr") or [None, None]
            st.markdown(f"Claimed **{pkr(ev.get('claimed_amount'))}** vs typical "
                        f"**{pkr(rng[0])} – {pkr(rng[1])}** for the damage in the photo "
                        f"(ratio {ev.get('cost_ratio', '—')}x of the typical maximum)")
            st.caption(ev.get("description_note", ""))

    with st.expander("🔎 What each agent returned (full output)"):
        st.caption("Each agent's exact output, as saved in the claim's checkpoint. "
                   "The note above is built only from these.")
        names = ["intake", "coverage", "evidence", "fraud", "decision", "critic"]
        for tab, name in zip(st.tabs([n.capitalize() for n in names]), names):
            with tab:
                st.markdown(f"**How it works:** {svc.AGENT_HOW[name]}")
                st.json(d["raw"][name], expanded=2)

    st.divider()
    with st.form(f"decide_{cid}"):
        st.markdown("#### Your decision")
        rec = d["recommendation"]
        options = ["accept", "approve", "investigate", "deny"]
        labels = {"accept": f"Accept the recommendation ({rec})", "approve": "Approve",
                  "investigate": "Send to investigation", "deny": "Deny"}
        action = st.radio("Decision", options, format_func=labels.get, horizontal=True,
                          label_visibility="collapsed")
        note = st.text_area("Note (recommended when you override the AI)", height=80)
        if st.form_submit_button("Submit decision", type="primary"):
            try:
                result = svc.decide(graph, cid, action, note=note, reviewer=reviewer)
                v = result["values"]
                word = "OVERRIDE" if v["human"]["overridden"] else "accepted"
                st.session_state["flash"] = f"{cid}: {v['final_decision'].upper()} ({word}) by {v['human']['reviewer']}"
                st.rerun()
            except ValueError as e:
                st.error(str(e))


def show_documents_pause(cid):
    pause = svc.claim_state(graph, cid)["pause"]
    st.warning(f"{cid}: documents incomplete. Missing: {', '.join(pause.get('missing_fields') or []) or '—'}")
    for issue in pause.get("issues") or []:
        st.markdown(f"- {issue}")
    c1, c2 = st.columns(2)
    if c1.button("Continue the review anyway", key=f"cont_{cid}"):
        svc.decide(graph, cid, "continue", reviewer=reviewer)
        st.rerun()
    if c2.button("Close the claim (ask the claimant)", key=f"close_{cid}"):
        svc.decide(graph, cid, "close", reviewer=reviewer)
        st.rerun()


with tab_queue:
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    m = st.columns(4)
    m[0].metric("Waiting for you", len(by_status.get("waiting_adjuster", [])))
    m[1].metric("Waiting for documents", len(by_status.get("waiting_documents", [])))
    m[2].metric("Decided", len(by_status.get("decided", [])))
    m[3].metric("Not processed", len(by_status.get("not_started", [])))

    waiting = by_status.get("waiting_adjuster", []) + by_status.get("waiting_documents", [])
    if not waiting:
        st.info("No claims are waiting. Process one in the **Process a claim** tab.")
    else:
        choice = st.selectbox("Claim to review", waiting, format_func=claim_label)
        st.divider()
        if choice["status"] == "waiting_documents":
            show_documents_pause(choice["claim"])
        else:
            show_review(choice["claim"])

# ---------------------------------------------------------------------------
# process a claim (live)
# ---------------------------------------------------------------------------
with tab_process:
    st.markdown("Run a claim through the agents. Coverage, evidence and fraud run **in parallel**; "
                "the decision agent waits for all three, the critic checks its note, and the claim then "
                "**pauses for you**.")
    todo = by_status.get("not_started", [])
    if not todo:
        st.info("Every dev claim has been processed. Use *Start over* below to run one again.")
    else:
        pick = st.selectbox("Claim", todo, format_func=claim_label, key="process_pick")
        if st.button(f"Process {pick['claim']}", type="primary"):
            lines = []
            with st.status(f"Processing {pick['claim']}...", expanded=True) as status:
                paused = None
                for ev in svc.process(graph, pick["claim"], use_llm=use_llm):
                    if ev["node"] == "__pause__":
                        paused = ev["pause"]
                    else:
                        lines.append(f"✅ **{ev['label']}** — {ev['summary']}")
                        st.write(lines[-1])
                if paused and paused.get("type") == "adjuster_review":
                    label = (f"{pick['claim']}: AI recommends {paused['recommendation'].upper()} "
                             f"— now in the Review queue")
                else:
                    label = f"{pick['claim']}: documents incomplete — now in the Review queue"
                status.update(label=label, state="complete", expanded=True)
            # The queue tab was drawn before this claim was processed, so reload the page once
            # (keeping this log) and the new claim appears in the Review queue straight away.
            st.session_state["last_run"] = {"label": label, "lines": lines}
            st.rerun()

    last = st.session_state.get("last_run")
    if last:
        with st.status(last["label"], state="complete", expanded=True):
            for line in last["lines"]:
                st.write(line)

    st.divider()
    done = [r for r in rows if r["status"] != "not_started"]
    with st.expander("Start over (forget a processed claim and its audit trail)"):
        if done:
            again = st.selectbox("Claim", done, format_func=claim_label, key="again_pick")
            sure = st.checkbox("Yes, delete this claim's saved run and audit trail")
            if st.button("Start over", disabled=not sure):
                svc.start_over(graph, again["claim"])
                st.rerun()
        else:
            st.caption("Nothing processed yet.")

# ---------------------------------------------------------------------------
# audit trail
# ---------------------------------------------------------------------------
with tab_audit:
    done = [r for r in rows if r["status"] != "not_started"]
    if not done:
        st.info("No claim has been processed yet.")
    else:
        table = [{"claim": r["claim"], "claimant": r["claimant"], "status": STATUS_TEXT[r["status"]],
                  "AI recommendation": r["recommendation"], "final decision": r["final_decision"],
                  "override": "yes" if r["overridden"] else ""} for r in done]
        st.dataframe(table, hide_index=True, width="stretch")
        pick = st.selectbox("Claim", done, format_func=claim_label, key="audit_pick")
        d = svc.details(svc.claim_state(graph, pick["claim"])["values"])
        h = d["human"]
        if d["final_decision"]:
            text = (f"Final decision **{d['final_decision'].upper()}** by **{h.get('reviewer')}** — "
                    + ("**overrode** the AI's recommendation" if h.get("overridden") else "accepted the recommendation")
                    + f" ({d['recommendation']}).")
            (st.warning if h.get("overridden") else st.success)(text)
            if h.get("note"):
                st.markdown(f"Adjuster's note: *{h['note']}*")
        st.dataframe([{"time": a["time"], "step": svc.STEP_LABELS.get(a["node"], a["node"]),
                       "what happened": a["summary"]} for a in d["audit"]],
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
with tab_results:
    res = svc.results()
    tests, comb = res["tests"], res["combined"]
    st.markdown("### Measured on claims the system never saw")
    st.caption("Each held-out set was run once, with the system frozen and the decision rules written down "
               "before the run. Nothing was tuned on these claims.")
    if "test" in tests or comb:
        c1, c2 = st.columns(2)
        if "test" in tests:
            s1 = tests["test"]["summary"]
            with c1.container(border=True):
                st.markdown("**Version 1** · 10 held-out claims")
                a, b = st.columns(2)
                a.metric("Accuracy", f"{s1['accuracy']:.0%}")
                b.metric("Wrong payouts", f"{s1['wrong_approvals']} / {s1['claims']}")
        if comb:
            with c2.container(border=True):
                st.markdown(f"**Version 2** · {comb['claims']} new held-out claims (tests 2 + 3)")
                a, b = st.columns(2)
                a.metric("Accuracy", f"{comb['accuracy']:.0%}",
                         delta=(f"{(comb['accuracy'] - tests['test']['summary']['accuracy']) * 100:+.0f} pts"
                                if "test" in tests else None))
                b.metric("Wrong payouts", f"{comb['wrong_approvals']} / {comb['claims']}")
        if comb:
            st.markdown("**Version 2 by claim type**")
            st.dataframe([{"claim type": k, "correct": f"{v[0]} / {v[1]}"}
                          for k, v in sorted(comb["by_scenario"].items())], hide_index=True, width="content")
            st.markdown(f"Wrong denials: **{comb['wrong_denials']}** · honest claims held up: "
                        f"**{comb['honest_held_up']}**")
        st.info("Every remaining error comes from the vision model misjudging damage severity. "
                "That is why every claim ends with a human adjuster.")
    else:
        st.info("No held-out results found in outputs/.")
    if res["decisions"]:
        with st.expander(f"Decision log ({len(res['decisions'])} recorded decisions)"):
            for e in res["decisions"]:
                st.markdown(f"**{e.get('time', '')}** · {e.get('decision', '')}  \n"
                            f"*decided by {e.get('decided_by', '?')}*")
