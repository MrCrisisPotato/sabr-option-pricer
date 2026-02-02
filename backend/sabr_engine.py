import QuantLib as ql
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import alpha, norm
import matplotlib.pyplot as plt
try:
    import cupy as cp
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


# Pricing Engines

def black_scholes_price(S, K, T, r, sigma, otype):
    """The Standard Black-Scholes Model (Spot-based)."""
    if T <= 0: 
        return max(0, S - K) if otype == 'CE' else max(0, K - S)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if otype == 'CE':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def black_76_price(f, K, T, r, sigma, otype):
    """The Black-76 Model (Forward-based). Used as the pricing engine for SABR."""
    if T <= 0: 
        return max(0, f - K) if otype == 'CE' else max(0, K - f)
    
    d1 = (np.log(f / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount = np.exp(-r * T)
    
    if otype == 'CE':
        return discount * (f * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        return discount * (K * norm.cdf(-d2) - f * norm.cdf(-d1))


def cuda_sabr_monte_carlo_price(F0, K, T, r, alpha, beta, rho, nu, otype, paths=100000, steps=100):
    """
    TRUE SABR Monte Carlo Simulation using GPU acceleration.
    Simulates the full SABR stochastic volatility model:
    dF = σ * F^β * dW1
    dσ = ν * σ * dW2
    where dW1 and dW2 have correlation ρ
    """
    dt = T / steps
    sqrt_dt = cp.sqrt(dt)
    
    # Initialize arrays on GPU
    F = cp.ones(paths) * F0
    sigma = cp.ones(paths) * alpha
    
    # Generate correlated Brownian motions
    for _ in range(steps):
        # Generate independent standard normals
        z1 = cp.random.standard_normal(paths)
        z2 = cp.random.standard_normal(paths)
        
        # Create correlation
        dW1 = z1 * sqrt_dt
        dW2 = (rho * z1 + cp.sqrt(1 - rho**2) * z2) * sqrt_dt
        
        # updating forward price using Euler-Maruyama scheme
        F = F + sigma * cp.power(cp.maximum(F, 1e-10), beta) * dW1
        
        # update volatility
        sigma = cp.maximum(sigma + nu * sigma * dW2, 1e-6)
        
        # Absorbing barrier at zero for forward prices
        F = cp.maximum(F, 0)
    
    # Calculate payoff
    if otype == 'CE':
        payoff = cp.maximum(F - K, 0)
    else:
        payoff = cp.maximum(K - F, 0)
    
    # Discount and average
    price = cp.exp(-r * T) * cp.mean(payoff)
    
    return float(price)


# Black-Scholes Greeks

def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def bs_delta(S, K, T, r, sigma, otype):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if otype == 'CE' else norm.cdf(d1) - 1





# Quantlib Helpers

def get_ql_option_type(otype):
    """Convert option type string to QuantLib type."""
    return ql.Option.Call if otype == 'CE' else ql.Option.Put


def implied_vol_finder(market_price, S, K, T, r, otype):
    """Back-calculates Implied Volatility from market price."""
    try:
        qtype = get_ql_option_type(otype)
        payoff = ql.PlainVanillaPayoff(qtype, K)
        ref_date = ql.Date().todaysDate()
        ql.Settings.instance().evaluationDate = ref_date
        
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(S)),
            ql.YieldTermStructureHandle(ql.FlatForward(ref_date, 0.0, ql.Actual365Fixed())),
            ql.YieldTermStructureHandle(ql.FlatForward(ref_date, r, ql.Actual365Fixed())),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ref_date, ql.NullCalendar(), 0.2, ql.Actual365Fixed()))
        )
        
        exercise = ql.EuropeanExercise(ref_date + int(T * 365))
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        
        return option.impliedVolatility(market_price, process)
    except:
        return np.nan  # Default fallback (old - 0.2)


def infer_spot_price_putcall_parity(group, r):
    """
    Improved spot price inference using Put-Call Parity:
    C - P = S - K*e^(-rT)
    S = C - P + K*e^(-rT)
    
    We find matching call-put pairs at the same strike and use their average.
    """
    T = (group['Expiry'].iloc[0] - group['Entry_Date'].iloc[0]).days / 365.0
    if T <= 0: 
        T = 1/365.0
    
    discount = np.exp(-r * T)
    
    # Pivot to get calls and puts side by side
    calls = group[group['Option type'] == 'CE'][['Strike Price', 'Entry_Premium']].rename(
        columns={'Entry_Premium': 'Call_Premium'})
    puts = group[group['Option type'] == 'PE'][['Strike Price', 'Entry_Premium']].rename(
        columns={'Entry_Premium': 'Put_Premium'})
    
    # Merge on strike price
    merged = pd.merge(calls, puts, on='Strike Price', how='inner')
    
    if len(merged) > 0:
        # Apply put-call parity: S = C - P + K*discount
        merged['Implied_S'] = merged['Call_Premium'] - merged['Put_Premium'] + merged['Strike Price'] * discount
        
        # Use median for robustness against outliers
        S = merged['Implied_S'].median()
    else:
        # Fallback: use ATM strike as rough estimate
        strikes = group['Strike Price'].values
        mid_strike = np.median(strikes)
        
        # For ATM options, premium ≈ intrinsic value for ITM
        # Use weighted average of call and put logic
        atm_call = group[(group['Option type'] == 'CE') & (group['Strike Price'] == mid_strike)]
        atm_put = group[(group['Option type'] == 'PE') & (group['Strike Price'] == mid_strike)]
        
        s_from_call = mid_strike + atm_call['Entry_Premium'].mean() if len(atm_call) > 0 else mid_strike
        s_from_put = mid_strike - atm_put['Entry_Premium'].mean() if len(atm_put) > 0 else mid_strike
        
        S = (s_from_call + s_from_put) / 2
    
    return S

# Processing and calibration

def process_day(group):
    """Process a single day's options data with calibration and pricing."""
    group = group.copy()
    # print("GROUP SIZE:", len(group))
    # print("ENTRY DATE:", group["Entry_Date"].iloc[0])

    r = 0.07  # Risk-free rate          
    try:
        S = group["Spot"].iloc[0]
    except:
        S = infer_spot_price_putcall_parity(group, r) # fallback
    
    T = (group['Expiry'].iloc[0] - group['Entry_Date'].iloc[0]).days / 365.0
    if T <= 0: 
        T = 1/365.0
    
    # Forward price for SABR
    fwd = S * np.exp(r * T)
    
    # Calculation of market implied vol
    group['Market_Vol'] = group.apply(
        lambda row: implied_vol_finder(
            row['Entry_Premium'], S, row['Strike Price'], T, r, row['Option type']
        ), axis=1
    )

    # Filter out extreme IVs
    group.loc[  
        (group['Market_Vol'] < 0.05) |
        (group['Market_Vol'] > 2.0),
        'Market_Vol'
    ] = np.nan

    valid = group['Market_Vol'].notna()

    strikes_fit = group.loc[valid, 'Strike Price'].values
    vols_fit = group.loc[valid, 'Market_Vol'].values
    print("SABR calibration points:", len(strikes_fit))
    print("IV range:", vols_fit.min(), vols_fit.max()) 

    if len(strikes_fit) < 5:
        raise ValueError("Not enough valid market IVs for SABR calibration")
    
    
    # SABR CALIBRATION
    # Initial guess: [Alpha, Rho, Nu]
    params = [0.15, -0.3, 0.3]
    beta = 0.7  # Fixed
    if len(strikes_fit) < 10:
        rho = 0.0
        params = [0.15, rho, 0.3]

    def objective_f(p):
        """Minimize RMSE between SABR and market volatilities."""
        
        alpha, rho, nu = p
        try:
            rho = np.clip(rho, -0.999, 0.999)
            vols = np.array([ql.sabrVolatility(k, fwd, T, alpha, beta, rho, max(nu, 1e-6)) for k in strikes_fit]) #this
            err = vols - vols_fit
            return np.sqrt(np.nanmean((err)**2))
        except:
            return 1e6
        
    
    # Constraints to ensure valid SABR parameters
    cons = (
        {'type': 'ineq', 'fun': lambda x: x[2]},         # Nu > 0
        {'type': 'ineq', 'fun': lambda x: x[0]},         # Alpha > 0
        {'type': 'ineq', 'fun': lambda x: 1 - abs(x[1])} # Rho in [-1, 1]
    )
    
    result = minimize(objective_f, params, constraints=cons, method='COBYLA')
    
    
    calibrated_params = result['x']
    alpha, rho, nu = calibrated_params
    
    # Pricing Comparison
    
    # SABR Implied Volatilities for each strike
    group['SABR_IV'] = np.nan
    group.loc[valid, 'SABR_IV'] = [ql.sabrVolatility(k, fwd, T, alpha, beta, rho, nu) for k in strikes_fit]   #this

    group['Smile_Shift'] = group['SABR_IV'] - group['Market_Vol']
    
    # SABR + Black-76 Pricing
    group['SABR_B76_Price'] = np.nan
    group.loc[valid, 'SABR_B76_Price'] = group.loc[valid].apply(
        lambda x: black_76_price(fwd, x['Strike Price'], T, r, x['SABR_IV'], x['Option type']), 
        axis=1
    )
    group['Mispricing_%'] = (
        (group['SABR_B76_Price'] - group['Entry_Premium'])
        / group['Entry_Premium']
    ) * 100
    
    # SABR + Monte Carlo Pricing (GPU Accelerated)
    if GPU_AVAILABLE:
        group['SABR_MC_Price'] = group.apply(
            lambda x: cuda_sabr_monte_carlo_price(
                fwd, x['Strike Price'], T, r, alpha, beta, rho, nu, x['Option type']
            ), 
            axis=1
        )
    
    # Black-Scholes Pricing with ATM vol
    atm_idx = (group['Strike Price'] - S).abs().idxmin()
    atm_vol = group.loc[atm_idx, 'Market_Vol']
    
    group['Vega'] = group.apply(
        lambda x: bs_vega(S, x['Strike Price'], T, r, atm_vol) * 0.01,
        axis=1
    )
    group['Delta'] = group.apply(
        lambda x: bs_delta(S, x['Strike Price'], T, r, atm_vol, x['Option type']),
        axis=1
    )
    group['BS_Price'] = group.apply(
        lambda x: black_scholes_price(S, x['Strike Price'], T, r, atm_vol, x['Option type']), #x['Market_Vol']
        axis=1
    )

    group['SABR_vs_BS'] = group['SABR_B76_Price'] - group['BS_Price']
    
    # Valuation Logic
    # Compare market premium against SABR Black-76 price
    group['Valuation'] = np.where(
        group['Entry_Premium'] < group['SABR_B76_Price'] * 0.95, 
        "Good (Cheap)", 
        np.where(
            group['Entry_Premium'] > group['SABR_B76_Price'] * 1.05, 
            "Bad (Expensive)", 
            "Fair"
        )
    )
    
    # Store calibrated parameters for reference
    group['SABR_Alpha'] = alpha
    group['SABR_Beta'] = beta
    group['SABR_Rho'] = rho
    group['SABR_Nu'] = nu
    group['Inferred_Spot'] = S
    
    return group


# Execution

if __name__ == "__main__":
    # Load data
    file_path = 'validated_option_pnl_time_analysis_30May2025.csv'
    df = pd.read_csv(file_path)
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
    df['Expiry'] = pd.to_datetime(df['Expiry'])
    
    # Process each day
    print("Processing options data with corrected models...")
    final_results = df.groupby('Entry_Date', group_keys=False).apply(process_day)
    
    # Output columns for display
    display_cols = [
        'Entry_Date', 'Option type', 'Strike Price', 'Entry_Premium', 
        'Market_Vol', 'SABR_IV', 'SABR_B76_Price', 'SABR_MC_Price', 
        'BS_Price', 'Valuation', 'Inferred_Spot'
    ]
    
    print("\n=== Sample Results ===")
    print(final_results[display_cols].head(20))
    
    # Save results
    output_file = 'corrected_model_comparison_results.csv'
    final_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    print("\n=== Pricing Model Comparison Summary ===")
    print(f"Average Market Premium: ${final_results['Entry_Premium'].mean():.2f}")
    print(f"Average SABR B76 Price: ${final_results['SABR_B76_Price'].mean():.2f}")
    print(f"Average SABR MC Price: ${final_results['SABR_MC_Price'].mean():.2f}")
    print(f"Average BS Price: ${final_results['BS_Price'].mean():.2f}")
    
    print("\n=== Valuation Distribution ===")
    print(final_results['Valuation'].value_counts())