import React from "react";

export default function OptionTable({ data }) {
  // 1️⃣ No data yet
  if (!data || data.length === 0) {
    return null;
  }

  // 2️⃣ Market closed / backend message
  if (!Array.isArray(data)) {
    return (
      <p style={{ color: "orange", marginTop: 20 }}>
        ⏳ {data.message || "Market closed or no live data available"}
      </p>
    );
  }

  // 3️⃣ Normal table render
  return (
    <table
      border="1"
      cellPadding="6"
      style={{ marginTop: 20, borderCollapse: "collapse" }}
    >
      <thead>
        <tr>
          <th>Strike</th>
          <th>Type</th>
          <th>Premium</th>
          <th>Market IV</th>
          <th>SABR IV</th>
          <th>SABR Price</th>
          <th>BS Price</th>
          <th>Mispricing %</th>
          <th>SABR − BS</th>
          <th>Vega</th>
          <th>Delta</th>
          <th>Valuation</th>
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i}>
            <td>{row["Strike Price"]}</td>
            <td>{row["Option type"]}</td>
            <td>{row["Entry_Premium"]}</td>
            <td>{row["Market_Vol"]?.toFixed(3)}</td>
            <td>{row["SABR_IV"]?.toFixed(3)}</td>
            <td>{row["SABR_B76_Price"]?.toFixed(2)}</td>
            <td>{row["BS_Price"]?.toFixed(2)}</td>
            <td style={{color: row["Mispricing_%"] > 10 ? "green" : row["Mispricing_%"] < -10 ? "red" : "black"}}>{row["Mispricing_%"]?.toFixed(1)}%</td>
            <td>{row["SABR_vs_BS"]?.toFixed(2)}</td>
            <td>{row["Vega"]?.toFixed(2)}</td>
            <td>{row["Delta"]?.toFixed(2)}</td>
            <td>{row["Valuation"]}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
