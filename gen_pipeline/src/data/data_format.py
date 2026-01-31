import json
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
from pathlib import Path


class Answer(BaseModel):
    name: str
    arguments: Dict[str, Any]


class QueryItem(BaseModel):
    query: str
    answers: List[Answer]

    @field_validator("answers")
    def check_answers_not_empty(cls, v):
        if not v:
            raise ValueError("answers list cannot be empty")
        return v


def parse_generated_data_safely(json_str) -> List[QueryItem]:
    """
    Parses the generated JSON format data and validates the format.
    """
    raw_data = json.loads(json_str)

    validated_data = [QueryItem(**item) for item in raw_data]
    return validated_data


def load_existing_data(file) -> List[QueryItem]:
    content = Path(file).read_text(encoding="utf-8")
    return parse_generated_data_safely(content)
