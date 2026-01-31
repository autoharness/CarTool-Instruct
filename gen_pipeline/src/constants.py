from pathlib import Path

METADATA_DIR = Path("gen_pipeline/metadata")
XLAM_FUNCTION_CALLING_SAMPLES_FILE = METADATA_DIR / "xlam_function_calling_samples.txt"
VEHICLE_PROPERTY_SCHEMA_FILE = METADATA_DIR / "vehicle_property_schema.txt"
VEHICLE_PROPERTIES_FILE = METADATA_DIR / "vehicle_properties.txt"
CAR_PROPERTY_FUNCTIONS_FILE = METADATA_DIR / "car_property_functions.txt"

SECRETS_PATH = Path("secrets/access_token.toml")
MODEL_NAME = "gemini-3-pro-preview"

FILTER_TOKENIZER_NAME = "gpt2"
FILTER_THRESHOLD = 0.8
