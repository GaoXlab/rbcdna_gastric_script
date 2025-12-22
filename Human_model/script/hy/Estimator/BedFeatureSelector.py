import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects top-n pre-sorted features from a 3-column BED file.

    Parameters
    ----------
    bed_path : str
        Path to the BED file (tab-separated, columns: chr, start, end).
    top_n : int, default=100
        Number of top features to select (takes first N rows of BED file).
    feature_name_format : str, default="{chr}:{start}-{end}"
        Format string to construct feature names from BED coordinates.
    """

    def __init__(self, bed_path, top_n=100, feature_name_format="chr{chr}:{start}-{end}"):
        self.bed_path = bed_path
        self.top_n = top_n
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

        # Take first top_n rows (assumes BED is pre-sorted)
        if self.top_n >= 0:
            top_bed = bed_df.head(self.top_n)
        else:
            top_bed = bed_df
        # Generate feature names
        self.selected_features_ = [
            self.feature_name_format.format(
                chr=row["chr"],
                start=row["start"],
                end=row["end"]
            )
            for _, row in top_bed.iterrows()
        ]
        available_features = set(X.columns)
        self.selected_features_ = [f for f in self.selected_features_ if f in available_features]
        # Check feature availability
        if hasattr(X, 'columns'):
            available_features = set(X.columns)
            self.selected_features_ = [f for f in self.selected_features_ if f in available_features]
        if len(self.selected_features_) != self.top_n and self.top_n >= 0:
            print(len(self.selected_features_))
            raise ValueError("bed和数据集中的特征数量不匹配，请检查bed文件和数据集的特征名称。")
        return self

    def transform(self, X):
        """Return only the selected features."""
        return X[self.selected_features_]


    def get_feature_names_out(self, input_features=None):
        """Return names of selected features (sklearn >= 1.0)."""
        return self.selected_features_
