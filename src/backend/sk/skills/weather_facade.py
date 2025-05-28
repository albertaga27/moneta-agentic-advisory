from pydantic import BaseModel, Field
import json
import os
import logging
from typing import List, Annotated, Optional
import requests
from requests_html import HTMLSession

from semantic_kernel.functions import kernel_function

class WeatherFacade:
    def __init__(self):
        pass  


    @kernel_function(
        name="get_weather", 
        description="Search for weather forecast data"
    )
    def get_weather(self) -> Annotated[str, "The output in JSON format"]:
        """
        Search the weather forecast, return a json.

        Returns:
        a JSON object with the weather forecast information found.

        """
        try:        
            location = "Switzerland"
            url = f"https://wttr.in/{location}?2nq&format=j1"
           
            response = requests.get(url)
            response.raise_for_status()  # raise an HTTPError if the request returned an unsuccessful status code

            if response.status_code == 200:
                return json.dumps(response.json())
            else:
                return {"error": f"HTTP {response.status_code}"}
        
        except Exception as e:
            logging.error(f"An unexpected error occurred in the 'get_weather' function of the 'weather_agent': {e}") 
            return 
        