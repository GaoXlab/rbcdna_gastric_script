import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ScoreOrderFilter(BaseEstimator, TransformerMixin):
    def __init__(self, bed_path, feature_name_format="chr{chr}:{start}-{end}"):
        self.bed_path = bed_path
        self.feature_name_format = feature_name_format
        self.selected_features_ = None

    def fit(self, X, y=None):
        """Load BED file and store top-n feature names."""
        # Read 3-column BED file (chr, start, end)
        bed_df = pd.read_csv(
            self.bed_path,
            sep="\t",
            header=None,
            usecols=[0, 1, 2],
            names=["chr", "start", "end"]
        )

        # Generate feature names
        self.selected_features_ = [
            self.feature_name_format.format(
                chr=row["chr"],
                start=row["start"],
                end=row["end"]
            )
            for _, row in bed_df.iterrows()
        ]
        # Check feature availability
        if hasattr(X, 'columns'):
            available_features = set(X.columns)
            self.selected_features_ = [f for f in self.selected_features_ if f in available_features]

        return self

    def transform(self, X):
        """Return only the selected features."""
        return X[self.selected_features_]


    def get_feature_names_out(self, input_features=None):
        """Return names of selected features (sklearn >= 1.0)."""
        return self.selected_features_
