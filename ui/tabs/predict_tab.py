import streamlit as st

from ui import api_client

ACTIONS = {"show_offer": "Show an offer", "no_action": "No action"}
VISITORS = ["Returning_Visitor", "New_Visitor", "Other"]


def _form() -> dict | None:
    with st.form("session"):
        c1, c2, c3 = st.columns(3)
        pages = c1.number_input("Product pages viewed", 0, 2000, 30)
        seconds = c1.number_input("Seconds on product pages", 0.0, value=1200.0, step=60.0)
        page_values = c1.number_input("PageValues", 0.0, value=0.0, step=0.5,
                                      help="Analytics metric; the model leans on it heavily.")
        exit_rate = c2.slider("Exit rate", 0.0, 1.0, 0.03, 0.005)
        bounce = c2.slider("Bounce rate", 0.0, 1.0, 0.01, 0.005)
        special = c2.selectbox("Special day closeness", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        visitor = c3.selectbox("Visitor type", VISITORS)
        weekend = c3.checkbox("Weekend")
        admin = c3.number_input("Administrative pages", 0, 100, 0)
        info = c3.number_input("Informational pages", 0, 100, 0)
        with st.expander("More details"):
            d1, d2 = st.columns(2)
            admin_s = d1.number_input("Administrative seconds", 0.0, value=0.0)
            info_s = d1.number_input("Informational seconds", 0.0, value=0.0)
            os_, browser = d2.number_input("Operating system code", 1, 20, 2), d2.number_input("Browser code", 1, 20, 2)
            region, traffic = d2.number_input("Region code", 1, 20, 1), d2.number_input("Traffic type code", 1, 30, 2)
        if not st.form_submit_button("Predict", type="primary"):
            return None
    return {"Administrative": admin, "Administrative_Duration": admin_s,
            "Informational": info, "Informational_Duration": info_s,
            "ProductRelated": pages, "ProductRelated_Duration": seconds,
            "BounceRates": bounce, "ExitRates": exit_rate, "PageValues": page_values,
            "SpecialDay": special, "Weekend": weekend, "OperatingSystems": os_,
            "Browser": browser, "Region": region, "TrafficType": traffic,
            "VisitorType": visitor}


def _show(result: dict) -> None:
    left, right = st.columns(2)
    left.metric("Purchase probability", f"{result['probability']:.1%}")
    right.metric("Recommended action", ACTIONS.get(result["action"], result["action"]))
    reasons = result.get("reasons")
    if not reasons:
        st.caption("Explanations are unavailable, so only the prediction is shown.")
        return
    st.subheader("Why this score")
    st.caption("Green raises the chance of a purchase. Red lowers it. Longer means stronger.")
    import matplotlib.pyplot as plt

    reasons = sorted(reasons, key=lambda r: r["effect"])
    labels = [f"{r['feature']} = {r['value']}" for r in reasons]
    colors = ["#2a9d8f" if r["effect"] > 0 else "#e63946" for r in reasons]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(labels, [r["effect"] for r in reasons], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pushes the score down  <--   -->  Pushes the score up", fontsize=13)
    ax.tick_params(labelsize=14)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

def render() -> None:
    payload = _form()
    if payload is None:
        return
    try:
        try:
            result = api_client.explain(payload)
        except api_client.ApiError:
            result = api_client.predict(payload)  
    except api_client.ApiError as err:
        st.error(str(err))
        return
    _show(result)