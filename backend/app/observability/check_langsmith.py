import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

project_name = os.getenv(
    "LANGSMITH_PROJECT",
    "Clinical-Trial-Site-Activation",
)

client = Client()

print("LangSmith project:", project_name)
print("Tracing enabled:", os.getenv("LANGSMITH_TRACING"))

# Send a small test trace to the configured project.
from langsmith import traceable

@traceable(name="langsmith_project_test")
def test_trace():
    return {"status": "ok", "project": project_name}

print(test_trace())
print("Test trace submitted. Check the LangSmith project list.")