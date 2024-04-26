from core.config import settings
from sqlalchemy import create_engine

engine = create_engine(
    str(settings.POSTGRES_URI), 
    echo=settings.POSTGRES_ECHO,
    pool_size=max(5, settings.POSTGRES_POOL_SIZE),
)
