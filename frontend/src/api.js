import axios from "axios";

const BASE = "http://localhost:8000";

export const fetchExpiries = async (underlying) => {
  const r = await axios.get(`${BASE}/expiries`, {
    params: { underlying }
  });
  return r.data;
};

export const fetchPrices = async (underlying, expiry) => {
  const r = await axios.get(`${BASE}/price`, {
    params: { underlying, expiry }
  });
  return r.data;
};
