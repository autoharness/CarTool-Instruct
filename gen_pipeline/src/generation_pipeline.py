import json
import random
import re
import tomllib
import logging
from typing import Dict, List, Optional
from constants import (
    SECRETS_PATH,
    MODEL_NAME,
    XLAM_FUNCTION_CALLING_SAMPLES_FILE,
    VEHICLE_PROPERTY_SCHEMA_FILE,
    VEHICLE_PROPERTIES_FILE,
    CAR_PROPERTY_FUNCTIONS_FILE,
)
from prompt.prompt import (
    GENERATION_PROMPT,
    SEED_GENERATION_EXAMPLE_SECTION,
    EXPANSION_GENERATION_EXAMPLE_SECTION,
)
from inference.api.llm_engine import LLMEngine, LLMOptions
from inference.gemini.google_gen_ai_engine import GoogleGenAIEngine
from data.data_format import QueryItem, parse_generated_data_safely, load_existing_data
from data.data_filter import QueryFilter

logger = logging.getLogger(__name__)


class GenerationPipeline:

    def __init__(self, num_samples, output, save_interval):
        self.num_samples = num_samples
        self.output_path = output
        self.save_interval = save_interval
        self.engine = self._initialize_engine()
        self.prompt_assets = self._load_prompt_assets()

    def _initialize_engine(self) -> LLMEngine:
        if not SECRETS_PATH.exists():
            raise FileNotFoundError(f"Secrets file not found at: {self.SECRETS_PATH}")

        with open(SECRETS_PATH, "rb") as f:
            secrets = tomllib.load(f)

        api_key = secrets.get("ai_services", {}).get("gemini", {}).get("api_key")
        if not api_key:
            raise ValueError("API key not found in secrets file.")

        engine = GoogleGenAIEngine(api_key=api_key)
        engine.load(
            LLMOptions(model_name=MODEL_NAME, temperature=1.0, thinking_mode=True)
        )
        return engine

    def _load_prompt_assets(self) -> Dict[str, str]:
        """Reads all static text files required for prompt construction."""
        xlam_samples = XLAM_FUNCTION_CALLING_SAMPLES_FILE.read_text(encoding="utf-8")
        vehicle_schema = VEHICLE_PROPERTY_SCHEMA_FILE.read_text(encoding="utf-8")

        raw_vehicle_props = VEHICLE_PROPERTIES_FILE.read_text(encoding="utf-8")
        raw_car_funcs = CAR_PROPERTY_FUNCTIONS_FILE.read_text(encoding="utf-8")

        # Validate and Minify
        try:
            props_obj = json.loads(raw_vehicle_props)
            vehicle_props = json.dumps(
                props_obj, separators=(",", ":"), ensure_ascii=False
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in vehicle properties metadata: {e}")

        try:
            funcs_obj = json.loads(raw_car_funcs)
            car_funcs = json.dumps(funcs_obj, separators=(",", ":"), ensure_ascii=False)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON format in car property functions metadata: {e}"
            )

        return {
            "xlam_samples": xlam_samples,
            "vehicle_schema": vehicle_schema,
            "vehicle_props": vehicle_props,
            "car_funcs": car_funcs,
        }

    def run(self):
        """
        Executes the generation pipeline: loads initial data (from disk or seed)
        and expands it until the sample target is met.
        """
        if self._is_cold_start():
            logger.info("Starting cold generation (Seed phase)...")
            data = self._generate_valid_seed()
            self._save_data(data)
        else:
            logger.info("Loading existing data...")
            data = load_existing_data(self.output_path)

        filter = QueryFilter(data)
        loop_iteration = 0
        while len(data) < self.num_samples:
            loop_iteration += 1
            expansion_batch = self._run_expansion_generation(data)
            if expansion_batch:
                filter.extend_unique(expansion_batch)
            # Logging
            current_count = len(data)
            percent = min(100.0, (current_count / self.num_samples) * 100)
            logger.info(
                "Expansion Loop %d: %d/%d (%.1f%%) - %s",
                loop_iteration,
                current_count,
                self.num_samples,
                percent,
                "Batch added" if expansion_batch else "No data added",
            )
            # Saving checkpoint.
            if self.save_interval > 0 and loop_iteration % self.save_interval == 0:
                self._save_data(data)
                logger.info(
                    f"Checkpoint reached (Loop {loop_iteration}). Intermediate data saved"
                )
        self._save_data(data)
        logger.info(f"Process completed. Total samples saved: {len(data)}")

    def _is_cold_start(self) -> bool:
        return not self.output_path.exists()

    def _generate_valid_seed(self, max_retries: int = 5) -> List[QueryItem]:
        """Attempts to generate seed data with a retry limit."""
        for attempt in range(max_retries):
            seed_data = self._run_seed_generation()
            if seed_data:
                return seed_data
        raise RuntimeError(
            f"Failed to generate seed data after {max_retries} attempts."
        )

    def _run_seed_generation(self) -> Optional[List[QueryItem]]:
        """Prepares prompt and runs generation for the seed phase."""
        example_section = SEED_GENERATION_EXAMPLE_SECTION.replace(
            "{xlam_function_calling_samples_placeholder}",
            self.prompt_assets["xlam_samples"],
        )
        prompt = self._construct_prompt(example_section, batch_num=15)
        return self._generate_batch(prompt)

    def _run_expansion_generation(
        self, warm_data: List[QueryItem], example_num: int = 8
    ) -> Optional[List[QueryItem]]:
        """Prepares prompt and runs generation for the expansion phase."""
        # Select random examples from existing data
        sample_size = min(example_num, len(warm_data))
        example_subset = random.sample(warm_data, sample_size)
        example_json = json.dumps(
            [item.model_dump() for item in example_subset], indent=2, ensure_ascii=False
        )
        example_section = EXPANSION_GENERATION_EXAMPLE_SECTION.replace(
            "{expansion_function_calling_samples_placeholder}", example_json
        )
        prompt = self._construct_prompt(example_section, batch_num=30)
        return self._generate_batch(prompt)

    def _construct_prompt(self, example_section: str, batch_num: int) -> str:
        replacements = {
            "{section_examples_placeholder}": example_section,
            "{vehicle_property_schema_placeholder}": self.prompt_assets[
                "vehicle_schema"
            ],
            "{vehicle_properties_placeholder}": self.prompt_assets["vehicle_props"],
            "{car_property_functions_placeholder}": self.prompt_assets["car_funcs"],
            "{pair_number_placeholder}": str(batch_num),
        }

        prompt = GENERATION_PROMPT
        for placeholder, content in replacements.items():
            prompt = prompt.replace(placeholder, content)

        return prompt

    def _generate_batch(self, prompt: str) -> Optional[List[QueryItem]]:
        response = self.engine.generate(prompt)
        json_str = self._extract_json_str(response.text)

        if not json_str:
            logger.warning(
                "No JSON content found in response. Response: %s", response.text
            )
            return None

        try:
            return parse_generated_data_safely(json_str)
        except Exception as e:
            logger.exception("Failed to parse generated JSON")
            return None

    @staticmethod
    def _extract_json_str(llm_response: str) -> Optional[str]:
        """
        Extracts the JSON substring from an LLM response.

        Strategies:
        1. Look for markdown code blocks (```json ... ```).
        2. If no code block, look for the outermost list brackets [...].
        """
        if not llm_response:
            return None

        text = llm_response.strip()

        code_block_pattern = r"```(?:json)?\s*(.*?)```"
        match = re.search(code_block_pattern, text, re.DOTALL)

        if match:
            return match.group(1).strip()

        start_idx = text.find("[")
        end_idx = text.rfind("]")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1]

        return None

    def _save_data(self, data: List[QueryItem]):
        """
        Saves the list of QueryItems to the output file in JSON format.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = [item.model_dump() for item in data]

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
