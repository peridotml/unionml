from fastapi import FastAPI

from tests.integration.pyspark_app.quickstart import model

app = FastAPI()
model.serve(app)
