from StanceDetectionInterface import StanceDetectionInterface
from genai_functions import complete_request, get_stance_prompt


class GenAIStanceOneShot(StanceDetectionInterface):
    def __init__(self, config):
        super().__init__(config)
        self.prompt_name = config.get("STANCE_PROMPT_NAME", "default")

    def predict_stances(self, examples):
        prompts = [
            get_stance_prompt(example["content"], example["query"], self.prompt_name)
            for example in examples
        ]
        results = complete_request(prompts, debug=False)
        return [self.assign_stance(result) for result in results]

    def assign_stance(self, result):
        if result is None or "stance" not in result:
            return None
        stance = str(result["stance"]).strip().lower()
        if stance not in self.valid_stances:
            return None
        return stance
