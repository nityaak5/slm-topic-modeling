from StanceDetectionInterface import StanceDetectionInterface
from genai_functions import (
    complete_request,
    stance_zero_shot_cot_prompt,
)


class GenAIStanceZeroShotCoT(StanceDetectionInterface):
    def predict_stances(self, examples):
        reasoning_prompts = [
            stance_zero_shot_cot_prompt(example["content"], example["query"])
            for example in examples
        ]
        reasoning_results = complete_request(reasoning_prompts, debug=False)
        reasoning_texts = [self.extract_reasoning(result) for result in reasoning_results]
        self.latest_reasoning_texts = reasoning_texts

        final_prompts = [
            stance_zero_shot_cot_prompt(
                example["content"],
                example["query"],
                stance_reason=reasoning_texts[i],
            )
            for i, example in enumerate(examples)
        ]
        final_results = complete_request(final_prompts, debug=False)
        return [self.assign_stance(result) for result in final_results]

    def extract_reasoning(self, result):
        if result is None:
            return ""
        reasoning = result.get("reasoning", "")
        return str(reasoning).strip()

    def assign_stance(self, result):
        if result is None or "stance" not in result:
            return None
        stance = self.canonicalize_stance_label(result["stance"])
        if stance not in self.valid_stances:
            return None
        return stance
