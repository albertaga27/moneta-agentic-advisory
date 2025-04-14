import feedparser
import json
import logging

from semantic_kernel.functions import kernel_function

class EnergyNewsFacade:
    def __init__(self):
        pass  

    @kernel_function(
            name="get_irena_rss_articles", 
            description="Search for Renewable Energy related articles, news forecasts etc. from the web"
        )
    def get_irena_rss_articles(self):
        """
        Fetches and parses articles from the IRENA RSS feed.

        Parameters
        ----------
        rss_url : str
            The URL of the RSS feed to parse.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing 'title' and 'link'.
        """
        rss_url = "https://www.irena.org/iapi/rssfeed/News"

        try:
            feed = feedparser.parse(rss_url)
            articles = []

            for entry in feed.entries:
                title = entry.get("title", "")
                link = entry.get("link", "")
                description = entry.get("description","")
                pubDate = entry.get("pubDate", "")
                articles.append({"title": title, "link": link, "description": description, "date" : pubDate})

            # Convert the list of articles to a JSON string
            articles_json = json.dumps(articles, indent=4)
        except Exception as e:
            logging.error(f"An unexpected error occurred in the 'get_irena_rss_articles' function of the 'energy_news_agent': {e}") 
            return 

        return articles_json    
    

    @kernel_function(
            name="get_today_energy_news", 
            description="Search for Today generic Energy related articles, news forecasts etc. from the web"
        )
    def get_today_energy_news(self):
        """
        Fetches and parses articles about Today Energy news from the EIA RSS feed.

        Parameters
        ----------
        rss_url : str
            The URL of the RSS feed to parse.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing 'title' and 'link'.
        """
        rss_url = "https://www.eia.gov/rss/todayinenergy.xml"

        try:
            feed = feedparser.parse(rss_url)
            articles = []

            for entry in feed.entries:
                print(f"\n Feed entry: {entry}")
                title = entry.get("title", "")
                link = entry.get("link", "")
                summary = entry.get("summary","")
                articles.append({"title": title, "link": link, "summary": summary})

            # Convert the list of articles to a JSON string
            articles_json = json.dumps(articles, indent=4)
        except Exception as e:
            logging.error(f"An unexpected error occurred in the 'get_today_energy_news' function of the 'energy_news_agent': {e}") 
            return 

        return articles_json    


