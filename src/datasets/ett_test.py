import pandas as pd


class ETTLoader:

    def __init__(
        self,
        csv_path
    ):

        self.csv_path = csv_path

    def load(self):

        df = pd.read_csv(
            self.csv_path
        )

        return df

    def get_features(
        self,
        drop_date=True
    ):

        df = self.load()

        if drop_date and "date" in df.columns:

            df = df.drop(
                columns=["date"]
            )

        return df.values
