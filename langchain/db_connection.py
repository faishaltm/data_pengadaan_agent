from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine


def create_db_connection(db_name):
    engine = create_engine(f"sqlite:///{db_name}")
    db = SQLDatabase(engine=engine)
    return db