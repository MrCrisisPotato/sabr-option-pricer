import React, { useEffect, useState } from "react";
import { fetchPrices, fetchExpiries } from "./api";
import OptionTable from "./OptionTable";

export default function App() {
  const [index, setIndex] = useState("NIFTY");
  const [expiries, setExpiries] = useState([]);
  const [expiry, setExpiry] = useState("");
  const [data, setData] = useState([]);
  const [error, setError] = useState("");
  const [underlying, setUnderlying] = useState("NIFTY");

  useEffect(() => {
    if (!underlying) return;
    fetchExpiries(underlying)
      .then(setExpiries)
      .catch(() => setError("Failed to load expiries"));
  }, [underlying]);

  async function load() {
    try {
      setError("");
      const res = await fetchPrices(underlying, expiry);
      setData(res);
    } catch (e) {
      const msg =
        e.response?.data?.detail
          ? String(e.response.data.detail)
          : "Failed to load prices";
      setError(msg);
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>SABR Option Pricer</h2>
      <select value={underlying} onChange={(e) => {setUnderlying(e.target.value); setExpiry(""); setData([]);}}>
        <option value="NIFTY">NIFTY 50</option>
        <option value="BANKNIFTY">BANK NIFTY</option>
        <option value="FINNIFTY">FIN NIFTY</option>
        <option value="MIDCPNIFTY">MIDCAP NIFTY</option>
      </select>
      <select
        value={expiry}
        onChange={(e) => setExpiry(e.target.value)}
      >
        <option value="">Select Expiry</option>
        {expiries.map((e) => (
          <option key={e} value={e}>{e}</option>
        ))}
      </select>

      <button disabled={!expiry} onClick={load}>
        Load Options
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      <OptionTable data={data} />
    </div>
  );
}
