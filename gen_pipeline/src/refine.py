import logging
import json
import random
import copy
from pathlib import Path
from typing import List, Dict
from data.data_format import load_existing_data
from constants import (
    VEHICLE_PROPERTY_SCHEMA_FILE,
    VEHICLE_PROPERTIES_FILE,
    CAR_PROPERTY_FUNCTIONS_FILE,
)
from prompt.prompt import DEVELOPER_MESSAGE_PROMPT

logger = logging.getLogger(__name__)


class PostProcess:
    """
    Handles the preparation of training and evaluation datasets.
    """

    def __init__(self, data_file, num_test, output_file):
        self.data_file = data_file
        self.num_test = num_test
        self.output_file = output_file

    def refine(self):
        """Orchestrates the data loading, splitting, formatting, and saving."""
        data_items = load_existing_data(self.data_file)
        total_count = len(data_items)
        if total_count < self.num_test:
            raise ValueError(
                f"Insufficient data items. Total found: {total_count}, "
                f"but requested num_test: {self.num_test}."
            )

        raw_tools_data = self._load_json_file(CAR_PROPERTY_FUNCTIONS_FILE)
        raw_vehicle_props = self._load_json_file(VEHICLE_PROPERTIES_FILE)

        base_prompt_template = DEVELOPER_MESSAGE_PROMPT.replace(
            "{vehicle_property_schema_placeholder}",
            VEHICLE_PROPERTY_SCHEMA_FILE.read_text(encoding="utf-8"),
        )

        random.shuffle(data_items)
        eval_items = data_items[: self.num_test]
        train_items = data_items[self.num_test :]

        train_entries = [
            self._build_jsonl_entry(
                item, "train", raw_tools_data, raw_vehicle_props, base_prompt_template
            )
            for item in train_items
        ]
        eval_entries = [
            self._build_jsonl_entry(
                item, "eval", raw_tools_data, raw_vehicle_props, base_prompt_template
            )
            for item in eval_items
        ]

        all_entries = train_entries + eval_entries
        self._write_jsonl(self.output_file, all_entries)

        logger.info(
            f"Dataset created: {self.output_file} (Train: {len(train_entries)}, Eval: {len(eval_entries)})"
        )

    def _load_json_file(self, file_path: Path):
        content = file_path.read_text(encoding="utf-8")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _build_jsonl_entry(
        self, item, metadata_label, raw_tools, raw_props, base_prompt
    ):
        """
        Builds a single entry, creating a shuffled version of tools and props.
        """
        shuffled_tools_def = self._randomize_tools(raw_tools)
        formatted_tools = [{"function": tool_def} for tool_def in shuffled_tools_def]

        shuffled_props_data = self._randomize_vehicle_props(raw_props)
        shuffled_props_str = json.dumps(
            shuffled_props_data, separators=(",", ":"), ensure_ascii=False
        )

        dev_msg = base_prompt.replace(
            "{vehicle_properties_placeholder}", shuffled_props_str
        )

        tool_calls = []
        for answer in item.answers:
            tool_calls.append(
                {"function": {"name": answer.name, "arguments": answer.arguments}}
            )

        messages = [
            {"role": "developer", "content": dev_msg},
            {"role": "user", "content": item.query},
            {"role": "assistant", "tool_calls": tool_calls},
        ]

        return {
            "metadata": metadata_label,
            "tools": formatted_tools,
            "messages": messages,
        }

    def _randomize_tools(self, tools_data: List[Dict]) -> List[Dict]:
        """
        Shuffles the list of tools AND the order of properties within each tool.
        """
        tools = copy.deepcopy(tools_data)
        random.shuffle(tools)

        for tool in tools:
            params = tool.get("parameters", {})
            if "properties" in params and isinstance(params["properties"], dict):
                props_dict = params["properties"]
                keys = list(props_dict.keys())
                random.shuffle(keys)

                # Reconstruct dictionary in new order (Python 3.7+ guarantees insertion order)
                shuffled_props_dict = {k: props_dict[k] for k in keys}
                params["properties"] = shuffled_props_dict

        return tools

    def _randomize_vehicle_props(self, props_data: List[Dict]) -> List[Dict]:
        """
        Shuffles the list of vehicle properties AND the areaIdProfiles within them.
        """
        props = copy.deepcopy(props_data)
        random.shuffle(props)

        for prop in props:
            if "areaIdProfiles" in prop and isinstance(prop["areaIdProfiles"], list):
                random.shuffle(prop["areaIdProfiles"])

        return props

    def _write_jsonl(self, path: Path, entries: List[Dict]):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
