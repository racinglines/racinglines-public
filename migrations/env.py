"""Alembic environment: migrations are generated from racinglines.db.models."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from racinglines.db.config import database_url
from racinglines.db.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)
# an explicit URL (e.g. from `racinglines db --db ... init`) wins over $DATABASE_URL
config.set_main_option("sqlalchemy.url", config.get_main_option("sqlalchemy.url") or database_url())
target_metadata = Base.metadata


def run_migrations_offline():
    context.configure(url=config.get_main_option("sqlalchemy.url"), target_metadata=target_metadata,
                      literal_binds=True, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    engine = engine_from_config(config.get_section(config.config_ini_section, {}),
                                prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
