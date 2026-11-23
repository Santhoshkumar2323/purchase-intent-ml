TARGET = "Revenue"
SPLIT_COLUMN = "Month"

NUMERIC = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
BINARY = ["Weekend"]
CATEGORICAL = ["OperatingSystems", "Browser", "Region", "TrafficType", "VisitorType"]

ALL_COLUMNS = NUMERIC + BINARY + CATEGORICAL + [SPLIT_COLUMN, TARGET]