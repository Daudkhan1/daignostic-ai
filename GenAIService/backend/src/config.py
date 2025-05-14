import os


class AmazonS3Config:
    """
    Configuration class for Amazon S3 access

    Attributes:
      AMAZON_S3_REGION_NAME: str = os.getenv("AMAZON_S3_REGION_NAME", "")
      AMAZON_S3_ACCESS_KEY: str = os.getenv("AMAZON_S3_ACCESS_KEY", "")
      AMAZON_S3_SECRET_ACCESS_KEY: str = os.getenv("AMAZON_S3_SECRET_ACCESS_KEY", "")
    """

    REGION_NAME: str = os.getenv("AMAZON_S3_REGION_NAME", "")
    ACCESS_KEY: str = os.getenv("AMAZON_S3_ACCESS_KEY", "")
    SECRET_ACCESS_KEY: str = os.getenv("AMAZON_S3_SECRET_ACCESS_KEY", "")


class MongoDBConfig:
    """
    Configuration class for MongoDB settings.

    Attributes:
        URI (str): The MongoDB connection URI.
        DB (str): The MongoDB database name.
    """

    URI: str = os.getenv("MONGODB_URI", "")
    DB_NAME: str = os.getenv("MONGODB_DB_NAME", "")


class GeminiConfig:
    """
    Configuration class for Google Gemini.

    Attributes:
        API_KEY (str): API KEY for gemini flash model.
        MODEL_NAME (str): Model name to use from gemini.
    """

    API_KEY: str = os.getenv("GEMINI_API_KEY", "AIzaSyDGPlGgv4KqjMfVEhTet1Tev9KKXiyBme4")
    MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")
