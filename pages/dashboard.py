# dashboard.py - analytics & data pages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = "data/DataCoSupplyChainDataset_no_zeros.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    # parse dates if present
    if "order date (DateOrders)" in df.columns:
        df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
        df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
        df["order_year"] = df["order_date"].dt.year
    else:
        df["order_month"] = "Unknown"
        df["order_year"] = np.nan
    # ensure numeric
    for c in ["Order Item Quantity","Order Item Discount","Product Price","Sales","Order Item Total","Days for shipping (real)","Days for shipment (scheduled)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # ensure target
    if "Late_delivery_risk" not in df.columns or df["Late_delivery_risk"].nunique()<2:
        if "Days for shipping (real)" in df.columns and "Days for shipment (scheduled)" in df.columns:
            df["Late_delivery_risk"] = (df["Days for shipping (real)"]>df["Days for shipment (scheduled)"]).astype(int)
        else:
            if "Delivery Status" in df.columns:
                df["Late_delivery_risk"] = df["Delivery Status"].apply(lambda x: 0 if isinstance(x,str) and "delivered" in x.lower() else 1).astype(int)
            else:
                df["Late_delivery_risk"] = 0
    return df

def run_dashboard():
    st.title("📊 Dashboard")
    df = load_data()

    # Filters row
    st.markdown("### Filters")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        city = st.selectbox("City", ["All"] + sorted(df["Order City"].dropna().astype(str).unique().tolist()))
    with c2:
        state = st.selectbox("State", ["All"] + sorted(df["Order State"].dropna().astype(str).unique().tolist()))
    with c3:
        cat = st.selectbox("Category", ["All"] + sorted(df["Category Name"].dropna().astype(str).unique().tolist()))
    with c4:
        mode = st.selectbox("Shipping Mode", ["All"] + sorted(df["Shipping Mode"].dropna().astype(str).unique().tolist()))

    min_qty, max_qty = 0, int(df["Order Item Quantity"].max()) if "Order Item Quantity" in df.columns else (0,100)
    qty = st.slider("Max quantity", min_qty, max_qty, max_qty)

    # apply filters
    d = df.copy()
    if city != "All":
        d = d[d["Order City"].astype(str)==city]
    if state != "All":
        d = d[d["Order State"].astype(str)==state]
    if cat != "All":
        d = d[d["Category Name"].astype(str)==cat]
    if mode != "All":
        d = d[d["Shipping Mode"].astype(str)==mode]
    if "Order Item Quantity" in d.columns:
        d = d[d["Order Item Quantity"].fillna(0) <= qty]

    # KPI row
    k1,k2,k3,k4 = st.columns(4)
    with k1:
        st.metric("Total Orders", f"{len(d):,}")
    with k2:
        delayed = int(d["Late_delivery_risk"].sum()) if "Late_delivery_risk" in d.columns else 0
        pct = (delayed/len(d)*100) if len(d)>0 else 0
        st.metric("Delayed Orders", f"{delayed:,}", f"{pct:.1f}%")
    with k3:
        sales = d["Sales"].sum() if "Sales" in d.columns else (d["Order Item Total"].sum() if "Order Item Total" in d.columns else 0)
        st.metric("Total Sales", f"₹{sales:,.2f}")
    with k4:
        avg_days = d["Days for shipping (real)"].mean() if "Days for shipping (real)" in d.columns else None
        st.metric("Avg Shipping Days", f"{avg_days:.2f}" if avg_days is not None and not np.isnan(avg_days) else "N/A")

    st.markdown("---")
    # Charts: Delay by Mode and Cities
    c5,c6 = st.columns((2,3))
    with c5:
        st.subheader("Shipping Mode Delay %")
        if "Shipping Mode" in d.columns:
            m = d.groupby("Shipping Mode").agg(total=("Late_delivery_risk","count"), delayed=("Late_delivery_risk","sum")).reset_index()
            m["delay_pct"] = m["delayed"]/m["total"]*100
            fig = px.bar(m, x="Shipping Mode", y="delay_pct", title="Delay % by Shipping Mode", template="plotly_dark", color="delay_pct")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No shipping mode data.")
    with c6:
        st.subheader("Top Cities by Delay %")
        if "Order City" in d.columns:
            cs = d.groupby("Order City").agg(total=("Late_delivery_risk","count"), delayed=("Late_delivery_risk","sum")).reset_index()
            cs = cs[cs["total"]>=10]
            cs["delay_pct"] = cs["delayed"]/cs["total"]*100
            cs = cs.sort_values("delay_pct", ascending=False).head(15)
            fig2 = px.bar(cs, x="delay_pct", y="Order City", orientation="h", template="plotly_dark", color="delay_pct")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No city data.")

    st.markdown("---")
    # Monthly trend
    st.subheader("Monthly Orders & Delays")
    if "order_month" in d.columns:
        monthly = d.groupby("order_month").agg(orders=("Late_delivery_risk","count"), delayed=("Late_delivery_risk","sum")).reset_index().sort_values("order_month")
        figm = go.Figure()
        figm.add_trace(go.Bar(x=monthly["order_month"], y=monthly["orders"], name="Orders"))
        figm.add_trace(go.Line(x=monthly["order_month"], y=monthly["delayed"], name="Delayed", yaxis="y2"))
        figm.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(figm, use_container_width=True)
    else:
        st.info("No order date data.")

def show_data_page():
    st.title("📄 Data Preview")
    df = load_data()
    st.dataframe(df.head(300), use_container_width=True)
    st.markdown("### Columns & dtypes")
    st.dataframe(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))

