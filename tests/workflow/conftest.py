"""Pytest configuration for workflow tests."""

import os
import pytest
from typing import Dict, Any

def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--test-env",
        action="store",
        default="local",
        help="Test environment: local, replit, or ci"
    )
    parser.addoption(
        "--db-url",
        action="store",
        help="Database URL for testing"
    )
    parser.addoption(
        "--web-host",
        action="store",
        help="Web server host for testing"
    )

@pytest.fixture(scope="session")
def test_config(request) -> Dict[str, Any]:
    """Get environment-specific test configuration."""
    env = request.config.getoption("--test-env")
    
    if env == "local":
        return {
            "database_url": request.config.getoption("--db-url")
                or "sqlite+aiosqlite:///test.db",
            "web_host": request.config.getoption("--web-host")
                or "http://localhost:8000",
            "use_ssl": False,
            "cleanup_db": True
        }
    elif env == "replit":
        return {
            "database_url": os.getenv("REPLIT_DB_URL")
                or "sqlite+aiosqlite:///test.db",
            "web_host": f"https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co",
            "use_ssl": True,
            "cleanup_db": True
        }
    elif env == "ci":
        return {
            "database_url": "sqlite+aiosqlite:///test.db",
            "web_host": "http://localhost:8000",
            "use_ssl": False,
            "cleanup_db": True
        }
    else:
        raise ValueError(f"Unknown test environment: {env}")

@pytest.fixture(scope="session")
def db_url(test_config) -> str:
    """Get database URL for testing."""
    return test_config["database_url"]

@pytest.fixture(scope="session")
def web_host(test_config) -> str:
    """Get web server host for testing."""
    return test_config["web_host"]

@pytest.fixture(scope="session")
def ssl_context(test_config):
    """Get SSL context for testing if needed."""
    if not test_config["use_ssl"]:
        return None
        
    import ssl
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

@pytest.fixture(autouse=True)
async def cleanup_database(request, test_config, db_engine):
    """Clean up database after tests if configured."""
    yield
    
    if test_config["cleanup_db"]:
        async with db_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def web_client(aiohttp_client, web_host, ssl_context):
    """Create web client with appropriate configuration."""
    app = web.Application()
    setup_routes(app)  # Add your route setup function
    
    if web_host.startswith("https"):
        return await aiohttp_client(
            app,
            server_kwargs={"ssl_context": ssl_context}
        )
    return await aiohttp_client(app)

def pytest_configure(config):
    """Configure pytest for workflow tests."""
    # Register markers
    config.addinivalue_line(
        "markers",
        "db: mark test as requiring database access"
    )
    config.addinivalue_line(
        "markers",
        "web: mark test as requiring web server access"
    )
    config.addinivalue_line(
        "markers",
        "ws: mark test as requiring WebSocket access"
    )
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Configure asyncio
    import asyncio
    if os.name == "nt":  # Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy()) 