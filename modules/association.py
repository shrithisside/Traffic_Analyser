                        #Name: Shriwayanta Maiti, PRN: 20240802169
                        #Name: Harshal Nanekar, PRN: 20240802163


"""
Association Rule Mining using Apriori algorithm.
Finds patterns between violation types co-occurring in the same area/day.
"""

import pandas as pd
import numpy as np

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False


def build_transactions(df: pd.DataFrame, by: str = "violation_type") -> list:
    """
    Build transaction list grouped by date + area.
    Each transaction = set of violations that occurred on same day in same area.
    """
    if df.empty:
        return []

    df = df.copy()
    df["date_str"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    if by == "violation_type":
        transactions = (
            df.groupby(["date_str", "area"])["violation_type"]
            .apply(list)
            .tolist()
        )
    elif by == "area":
        transactions = (
            df.groupby(["date_str", "month"])["area"]
            .apply(list)
            .tolist()
        )
    else:
        transactions = (
            df.groupby(["date_str", "area"])["violation_type"]
            .apply(list)
            .tolist()
        )

    # Remove duplicates within each transaction, keep only non-empty
    transactions = [list(set(t)) for t in transactions if len(t) > 0]
    return transactions


def run_association_rules(
    df: pd.DataFrame,
    min_support: float = 0.05,
    min_confidence: float = 0.4,
    min_lift: float = 1.0,
    by: str = "violation_type",
) -> tuple:
    """
    Run Apriori + association rules.
    Returns: (frequent_itemsets_df, rules_df)
    """
    if not MLXTEND_AVAILABLE:
        return pd.DataFrame(), pd.DataFrame()

    transactions = build_transactions(df, by=by)

    if len(transactions) < 5:
        return pd.DataFrame(), pd.DataFrame()

    te = TransactionEncoder()
    te_array = te.fit_transform(transactions)
    trans_df = pd.DataFrame(te_array, columns=te.columns_)

    # Run Apriori
    frequent_itemsets = apriori(
        trans_df,
        min_support=min_support,
        use_colnames=True,
        max_len=3,
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    # Generate rules
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
        num_itemsets=len(frequent_itemsets),
    )

    # Filter by lift
    rules = rules[rules["lift"] >= min_lift].copy()

    # Clean up for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))
    rules = rules.round(4)

    freq_display = frequent_itemsets.copy()
    freq_display["itemsets"] = freq_display["itemsets"].apply(
        lambda x: ", ".join(list(x))
    )

    return freq_display, rules