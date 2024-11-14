from letta.main import app
from letta.server.rest_api.app import app as fastapi_app
import uvicorn

if __name__ == "__main__":
    # If you want to run the CLI version
    # app()

    # If you want to run the FastAPI server
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)