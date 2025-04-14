from pydantic import BaseModel, Field
import json
import os
import logging
from typing import List, Annotated, Optional
import requests
import pandas as pd
from requests_html import HTMLSession

from semantic_kernel.functions import kernel_function

class ElectricityFacade:
    def __init__(self):
        pass  


    @kernel_function(
        name="get_electricity_production", 
        description="Search for swiss grid electricity production data in terms of GigaWatt hour in a certain day for each energy category"
    )
    def get_electricity_production(self) -> Annotated[str, "The output in JSON format"]:
        """
        Search the web for swiss grid electricity production data into a pandas DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the data found Datum, Energietraeger, Produktion_GWh

        """
        try:
            # URL of the CSV file
            url = "https://www.uvek-gis.admin.ch/BFE/ogd/104/ogd104_stromproduktion_swissgrid.csv"

            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # raise an HTTPError if the request returned an unsuccessful status code

            # The CSV file content as text
            csv_content = response.text

            # Convert the CSV content into a Pandas DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content), sep=",")

            # Ensure the 'Datum' column is treated as dates.
            # Adjust the format string if your CSV uses a different date format.
            df['Datum'] = pd.to_datetime(df['Datum'], format="%Y-%m-%d", dayfirst=False)

            # Determine the last date in the dataset and compute an offset
            max_date = df['Datum'].max()
            start_date = max_date - pd.DateOffset(months=2)
            # Filter the DataFrame to rows in the last year 
            mask = (df['Datum'] >= start_date)
            df_filtered = df.loc[mask].copy()

            # Show the first few rows
            logging.debug(df_filtered.head())
            
            return df_filtered.to_json(orient="records", date_format="iso")
        except Exception as e:
            logging.error(f"An unexpected error occurred in the 'get_electricity_consumption' function of the 'electricity_agent': {e}") 
            return 
        
    @kernel_function(
        name="get_electricity_consumption", 
        description="Search for swiss grid electricity consumption data in terms of GigaWatt hour in a certain day consumed and needed (Landesverbrauch GWh, Endverbrauch GWh)"
    )
    def get_electricity_consumption(self) -> Annotated[str, "The output in JSON format"]:
        """
        Search the web for swiss grid electricity consumption data into a pandas DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the data found: Datum, Landesverbrauch GWh, Endverbrauch GWh

        """
        try:
            # URL of the CSV file
            url = "https://www.uvek-gis.admin.ch/BFE/ogd/103/ogd103_stromverbrauch_swissgrid_lv_und_endv.csv"

            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # raise an HTTPError if the request returned an unsuccessful status code

            # The CSV file content as text
            csv_content = response.text

            # Convert the CSV content into a Pandas DataFrame
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content), sep=",")

            # Ensure the 'Datum' column is treated as dates.
            # Adjust the format string if your CSV uses a different date format.
            df['Datum'] = pd.to_datetime(df['Datum'], format="%Y-%m-%d", dayfirst=False)

            # Determine the last date in the dataset and compute an offset
            max_date = df['Datum'].max()
            start_date = max_date - pd.DateOffset(months=2)
            # Filter the DataFrame to rows in the last year 
            mask = (df['Datum'] >= start_date)
            df_filtered = df.loc[mask].copy()

            # Show the first few rows
            logging.debug(df_filtered.head())
            
            return df_filtered.to_json(orient="records", date_format="iso")
        except Exception as e:
            logging.error(f"An unexpected error occurred in the 'get_electricity_consumption' function of the 'electricity_agent': {e}") 
            return 



    def filter_last_year_by_energietraeger(df, energietraeger):
        """
        Filters the given DataFrame to return only rows from the last year
        (based on the maximum date in the 'Datum' column) 
        and matching the specified 'Energietraeger'.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['Datum', 'Energietraeger', 'Produktion_GWh'].
        energietraeger : str
            The energy source to filter for (e.g., 'Kernkraft', 'Wind', etc.)

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame containing only the rows for the last year
            and the specified Energietraeger.
        """

        # Ensure the 'Datum' column is treated as dates.
        # Adjust the format string if your CSV uses a different date format.
        df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y')

        # Determine the last date in the dataset and compute a one-year offset
        max_date = df['Datum'].max()
        start_date = max_date - pd.DateOffset(years=1)

        # Filter the DataFrame to rows in the last year and matching Energietraeger
        mask = (df['Datum'] >= start_date) & (df['Energietraeger'] == energietraeger)
        df_filtered = df.loc[mask].copy()

        return df_filtered