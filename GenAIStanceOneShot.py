from StanceDetectionInterface import StanceDetectionInterface
from genai_functions import (
    complete_request,
    get_stance_prompt,
)


class GenAIStanceOneShot(StanceDetectionInterface):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_name = config.get("STANCE_PROMPT_NAME", "default")

    def predict_stances(self, examples):
        if self.prompt_name == "cot":
            reasoning_prompts = [
                get_stance_prompt(example["content"], example["query"], self.prompt_name)
                for example in examples
            ]
            reasoning_results = complete_request(reasoning_prompts, debug=False)
            reasoning_texts = [self.extract_reasoning(result) for result in reasoning_results]
            self.latest_reasoning_texts = reasoning_texts

            final_prompts = [
                get_stance_prompt(
                    example["content"],
                    example["query"],
                    self.prompt_name,
                    stance_reason=reasoning_texts[i],
                )
                for i, example in enumerate(examples)
            ]
            final_results = complete_request(final_prompts, debug=False)
            return [self.assign_stance(result) for result in final_results]

        prompts = [
            get_stance_prompt(example["content"], example["query"], self.prompt_name, doc_id=example.get("doc_id"))
            for example in examples
        ]
        results = complete_request(prompts, debug=False)
        return [self.assign_stance(result) for result in results]

    def extract_reasoning(self, result):
        if result is None:
            return ""
        reasoning = result.get("reasoning", "")
        return str(reasoning).strip()

    def assign_stance(self, result):
        if result is None:
            return None

        if "stance" in result:
            stance = self.canonicalize_stance_label(result["stance"])
            if stance in self.valid_stances:
                return stance

        # Experiment 2 path: map 5-point stance score to 3-way stance label.
        # Primary mapping:
        # 1, 2 -> AGAINST
        # 3    -> NONE
        # 4, 5 -> FAVOR
        # Safe backward compatibility:
        # -2, -1 -> AGAINST, 0 -> NONE
        score_keys = ("stance_score", "score", "stance_rating")
        for key in score_keys:
            if key not in result:
                continue
            mapped = self._map_stance_score_to_label(result.get(key))
            if mapped in self.valid_stances:
                return mapped
        return None

    def _map_stance_score_to_label(self, raw_score):
        if raw_score is None:
            return None

        score = None
        if isinstance(raw_score, (int, float)):
            score = int(raw_score)
        else:
            text = str(raw_score).strip()
            if text.startswith("+"):
                text = text[1:]
            try:
                score = int(text)
            except ValueError:
                return None

        if score in (-2, -1, 1, 2):
            return "AGAINST"
        if score in (3, 0):
            return "NONE"
        if score in (4, 5):
            return "FAVOR"
        return None
