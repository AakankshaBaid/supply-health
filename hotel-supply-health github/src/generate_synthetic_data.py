"""
generate_synthetic_data.py
---------------------------
Generates a synthetic hotel-supply dataset for the "Supply Performance
Early-Warning Model" portfolio project.

WHY SYNTHETIC DATA
This project is a public, sanitized reconstruction of a methodology I use
professionally in travel-marketplace supply analytics (identifying hotel
partners whose commercial performance is likely to lag their competitive
set). No production data, table schemas, partner identities, or company-
proprietary metrics are used anywhere in this repo. Every value below is
generated from a documented statistical process, seeded for reproducibility.

The data-generating process intentionally encodes realistic, correlated
relationships (and irreducible noise) so that the downstream modeling in
the notebook has to do genuine work -- it is not a toy dataset with a
trivial separating boundary.

Run:
    python src/generate_synthetic_data.py
Output:
    data/synthetic_hotel_supply_data.csv
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_ROWS = 150000

MARKET_SEGMENTS = ["Urban Business", "Leisure Beach", "Suburban", "Airport", "Resort/Destination"]
REGIONS = ["West", "Midwest", "South", "Northeast", "International"]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate(n_rows: int = N_ROWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hotel_id = [f"HTL-{i:05d}" for i in range(1, n_rows + 1)]
    market_segment = rng.choice(MARKET_SEGMENTS, size=n_rows, p=[0.28, 0.22, 0.24, 0.14, 0.12])
    region = rng.choice(REGIONS, size=n_rows, p=[0.24, 0.19, 0.22, 0.20, 0.15])

    # Star rating: skewed toward 3-4 star supply, as in most OTA catalogs
    star_rating = rng.choice([1, 2, 3, 4, 5], size=n_rows, p=[0.03, 0.10, 0.38, 0.36, 0.13])

    # Partner tenure
    days_listed = _clip(rng.gamma(shape=2.2, scale=260, size=n_rows), 15, 3650).astype(int)
    is_new_partner = (days_listed < 180).astype(int)

    # Review score correlates with star rating + noise
    review_score = _clip(
        0.55 * star_rating + rng.normal(1.6, 0.55, n_rows), 1.0, 5.0
    ).round(2)

    # Review count grows with tenure, noisy
    review_count = _clip(
        (days_listed / 6.0) * rng.gamma(2.0, 1.0, n_rows) / 2.0, 0, 4000
    ).astype(int)

    # Content completeness (0-100 composite of photos/description/amenities)
    photo_count = _clip(rng.normal(24, 11, n_rows), 3, 90).astype(int)
    content_completeness_score = _clip(
        0.55 * (photo_count / 90 * 100) + rng.normal(35, 14, n_rows), 5, 100
    ).round(1)
    has_virtual_tour = rng.binomial(1, p=_clip(0.15 + content_completeness_score / 400, 0, 0.6))

    # Commercial levers
    price_index_vs_compset = _clip(rng.normal(1.0, 0.18, n_rows), 0.55, 1.9).round(3)
    promo_depth_pct = _clip(rng.beta(2.0, 5.0, n_rows) * 45, 0, 45).round(1)
    inventory_availability_rate = _clip(rng.beta(5.0, 2.0, n_rows) * 100, 5, 100).round(1)

    # Guest experience / ops signals
    mobile_conversion_rate = _clip(rng.beta(2.0, 30.0, n_rows) * 100, 0.2, 15).round(2)
    guest_response_rate_pct = _clip(rng.beta(6.0, 1.6, n_rows) * 100, 10, 100).round(1)
    cancellation_rate_pct = _clip(rng.beta(2.2, 9.0, n_rows) * 100, 0, 45).round(1)
    days_since_last_content_update = _clip(rng.exponential(140, n_rows), 0, 900).astype(int)

    comp_set_size = rng.integers(5, 26, n_rows)

    # Snapshot month (1-12): gives the dataset a genuine time axis so
    # out-of-time validation has something real to test against, rather
    # than a random split masquerading as one.
    snapshot_month = rng.integers(1, 13, n_rows)

    # ---------------------------------------------------------------
    # Latent "true" data-generating process for the target.
    # Positive coefficients push toward UNDERPERFORMING (target = 1).
    # Standardize each driver before weighting so no single raw scale
    # dominates purely because of units.
    # ---------------------------------------------------------------
    def z(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-9)

    # Mild concept drift over the year: price sensitivity strengthens
    # slightly in later months (e.g. tightening travel budgets), so a
    # model trained only on early months should generalize slightly
    # worse to later months than a random split would suggest.
    drift = 0.05 * (snapshot_month - 6.5) * z(price_index_vs_compset)

    logit = (
        0.55 * z(price_index_vs_compset)          # pricier than comp set -> more likely underperforming
        - 0.60 * z(content_completeness_score)     # richer content -> less likely
        - 0.45 * z(inventory_availability_rate)    # more open-for-sale nights -> less likely
        - 0.35 * z(promo_depth_pct)                # deeper promo participation -> less likely
        - 0.50 * z(review_score)                   # better reviews -> less likely
        + 0.30 * z(cancellation_rate_pct)           # more cancellations -> more likely
        - 0.20 * z(guest_response_rate_pct)         # responsive partner -> less likely
        + 0.15 * z(days_since_last_content_update)  # stale content -> more likely
        + 0.25 * is_new_partner                      # new partners start behind
        - 0.10 * z(review_count)
        + drift                                       # mild seasonal concept drift
        + rng.normal(0, 1.0, n_rows)                 # irreducible noise
    )

    # Calibrate the intercept so the base rate lands close to real-world
    # imbalance seen in supply-health work (roughly 70/30).
    intercept = np.quantile(logit, 0.78)
    prob_underperforming = 1 / (1 + np.exp(-(logit - intercept)))
    target = rng.binomial(1, prob_underperforming)

    df = pd.DataFrame({
        "hotel_id": hotel_id,
        "snapshot_month": snapshot_month,
        "market_segment": market_segment,
        "region": region,
        "star_rating": star_rating,
        "days_listed": days_listed,
        "is_new_partner": is_new_partner,
        "review_score": review_score,
        "review_count": review_count,
        "photo_count": photo_count,
        "content_completeness_score": content_completeness_score,
        "has_virtual_tour": has_virtual_tour,
        "price_index_vs_compset": price_index_vs_compset,
        "promo_depth_pct": promo_depth_pct,
        "inventory_availability_rate": inventory_availability_rate,
        "mobile_conversion_rate": mobile_conversion_rate,
        "guest_response_rate_pct": guest_response_rate_pct,
        "cancellation_rate_pct": cancellation_rate_pct,
        "days_since_last_content_update": days_since_last_content_update,
        "comp_set_size": comp_set_size,
        "underperforming": target,
    })

    # ---------------------------------------------------------------
    # Realistic, MCAR-ish missingness to mirror real supply-data quality
    # issues (mirrors the data-quality audit step in the notebook).
    # ---------------------------------------------------------------
    review_missing_mask = rng.random(n_rows) < np.where(is_new_partner == 1, 0.22, 0.03)
    df.loc[review_missing_mask, "review_score"] = np.nan

    mobile_missing_mask = rng.random(n_rows) < 0.06
    df.loc[mobile_missing_mask, "mobile_conversion_rate"] = np.nan

    tour_missing_mask = rng.random(n_rows) < 0.04
    df.loc[tour_missing_mask, "has_virtual_tour"] = np.nan

    return df


if __name__ == "__main__":
    data = generate()
    out_path = "data/synthetic_hotel_supply_data.csv"
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data):,} rows to {out_path}")
    print(f"Base rate (underperforming=1): {data['underperforming'].mean():.1%}")
