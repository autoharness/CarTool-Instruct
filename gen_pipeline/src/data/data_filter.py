import logging
import re
from typing import List, Iterable
from .data_format import QueryItem
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from constants import FILTER_THRESHOLD, FILTER_TOKENIZER_NAME

logger = logging.getLogger(__name__)


class QueryFilter:
    """
    Filters data by rejecting those with high ROUGE-L token overlap.
    """

    MASK_TOKEN = "<ARG>"
    DEBUG = True

    def __init__(self, existing_data: List[QueryItem]):
        self.tokenizer = AutoTokenizer.from_pretrained(FILTER_TOKENIZER_NAME)
        self._existing_data = existing_data
        self._existing_tokens: List[List[str]] = [
            self._get_masked_tokens(item) for item in existing_data
        ]

    def extend_unique(self, new_data: List[QueryItem]):
        """
        Adds new items to the dataset only if they are distinct from existing data.
        """
        accepted_buffer = []
        accepted_tokens_buffer = []

        for candidate in new_data:
            candidate_tokens = self._get_masked_tokens(candidate)

            if not self._validate(candidate_tokens, self._existing_tokens):
                continue

            if not self._validate(candidate_tokens, accepted_tokens_buffer):
                continue

            accepted_buffer.append(candidate)
            accepted_tokens_buffer.append(candidate_tokens)

        self._existing_data.extend(accepted_buffer)
        self._existing_tokens.extend(accepted_tokens_buffer)

    def _validate(
        self, candidate_tokens: List[str], context_tokens: Iterable[List[str]]
    ) -> bool:
        for existing_tokens in context_tokens:
            score = rouge_scorer._score_lcs(candidate_tokens, existing_tokens).fmeasure
            if score > FILTER_THRESHOLD:
                if self.DEBUG:
                    c_s = self.tokenizer.convert_tokens_to_string(candidate_tokens)
                    e_s = self.tokenizer.convert_tokens_to_string(existing_tokens)
                    logger.debug(f"Query '{c_s}' overlaps '{e_s}'. Score: {score}")
                return False
        return True

    def _get_masked_tokens(self, item: QueryItem) -> List[str]:
        """
        Replaces specific argument values in the query with a generic mask
        and then tokenizes the result.
        """
        masked_query = item.query

        # Collect all values from arguments that need to be masked
        values_to_mask = []
        for answer in item.answers:
            for val in answer.arguments.values():
                # Only mask strings and numbers
                if isinstance(val, (str, int, float)):
                    values_to_mask.append(str(val))

        # Sort by length (descending) to handle substrings correctly
        # (e.g. avoid masking "New" inside "New York" leaving "[MASK] York")
        values_to_mask.sort(key=len, reverse=True)

        for val in values_to_mask:
            # We skip very short strings (like "a" or "I") to avoid destroying the sentence structure
            # But we always mask numbers (digits)
            if len(val) > 1 or val.isdigit():
                pattern = re.escape(val)
                masked_query = re.sub(
                    pattern, self.MASK_TOKEN, masked_query, flags=re.IGNORECASE
                )

        return self.tokenizer.tokenize(masked_query)
