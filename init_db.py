from alembic.config import Config
from alembic import command


def run_migrations():
    # Create Alembic configuration object
    alembic_cfg = Config("alembic.ini")

    try:
        # Run the migrations
        command.upgrade(alembic_cfg, "head")
        print("Database migrations completed successfully!")
    except Exception as e:
        print(f"Error running migrations: {e}")


if __name__ == "__main__":
    run_migrations()
