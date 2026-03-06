"""
Simple RAGAS Evaluation Script
"""

import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

input_path = Path(__file__).parent / "datasets" / "ragas_evaluation_dataset-1.json"
# Load your dataset with explicit UTF-8 encoding to handle special characters
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert to RAGAS format
dataset = Dataset.from_dict({
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
    "contexts": [item["contexts"] for item in data],
})

# Set up evaluator (using GPT-4 for evaluation)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Run evaluation
results = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness, 
        answer_relevancy, 
        # context_precision, 
        # context_recall
    ],
    llm=llm,
    embeddings=embeddings,
)

# Convert to DataFrame first
df = results.to_pandas()

# Save to CSV
df.to_csv("evaluation/datasets/results.csv", index=False)
print("\n✅ Detailed results saved to results.csv")