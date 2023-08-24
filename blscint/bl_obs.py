import requests
import pandas as pd
import pytz
import datetime


def check_btl():
    # Read public GBT schedule
    url = "https://dss.gb.nrao.edu/schedule/public"
    html = requests.get(url).content 
    df = pd.read_html(html)[0]

    # Limit to today's scheduling blocks
    end_idx = df[df["Project ID"] == "Project ID"].index[0]
    df = df.iloc[:end_idx]

    # Format dates
    df[["Begin_Date", "End_Date"]] = df.iloc[:, 0].str.split("-", expand=True).apply(lambda x: x.str.replace("\+| ", "", regex=True))

    # Get the Eastern time zone
    eastern = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern)

    # Iterate over times to check whether the BTL backend is being used
    for i, row in df.iterrows():
        begin = now.replace(hour=int(row["Begin_Date"][:2]), 
                            minute=int(row["Begin_Date"][-2:]),
                            second=0,
                            microsecond=0)
        end = now.replace(hour=int(row["End_Date"][:2]), 
                        minute=int(row["End_Date"][-2:]),
                        second=0,
                        microsecond=0)
        if int(row["End_Date"][:2]) < int(row["Begin_Date"][:2]):
            end += datetime.timedelta(days=1)

        if begin <= now < end:
            if row["Backends"] == "BTL":
                return True 
            else:
                return False