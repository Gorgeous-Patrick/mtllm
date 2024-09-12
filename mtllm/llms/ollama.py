"""Ollama client for MTLLM."""

from mtllm.llms.base import BaseLLM
import os
import json

REASON_SUFFIX = """
Reason and return the output result(s) only, adhering to the provided Type in the following format

[Reasoning] <Reason>
[Output] <Result>
"""

NORMAL_SUFFIX = """Generate and return the output result(s) only, adhering to the provided Type in the following format

[Output] <result>
"""  # noqa E501

CHAIN_OF_THOUGHT_SUFFIX = """
Generate and return the output result(s) only, adhering to the provided Type in the following format. Perform the operation in a chain of thoughts.(Think Step by Step)

[Chain of Thoughts] <Chain of Thoughts>
[Output] <Result>
"""  # noqa E501

REACT_SUFFIX = """
You are given with a list of tools you can use to do different things. To achieve the given [Action], incrementally think and provide tool_usage necessary to achieve what is thought.
Provide your answer adhering in the following format. tool_usage is a function call with the necessary arguments. Only provide one [THOUGHT] and [TOOL USAGE] at a time.

[Thought] <Thought>
[Tool Usage] <tool_usage>
"""  # noqa E501


class Ollama(BaseLLM):
    """Ollama API client for Large Language Models (LLMs)."""

    MTLLM_METHOD_PROMPTS: dict[str, str] = {
        "Normal": NORMAL_SUFFIX,
        "Reason": REASON_SUFFIX,
        "Chain-of-Thoughts": CHAIN_OF_THOUGHT_SUFFIX,
        "ReAct": REACT_SUFFIX,
    }

    def __init__(
        self,
        verbose: bool = False,
        max_tries: int = 10,
        type_check: bool = False,
        **kwargs: dict
    ) -> None:
        """Initialize the Ollama API client."""
        import ollama  # type: ignore

        super().__init__(verbose, max_tries, type_check)
        self.client = ollama.Client(host=kwargs.get("host", "http://localhost:11434"))
        self.model_name = kwargs.get("model_name", "phi3")
        self.default_model_params = {
            k: v for k, v in kwargs.items() if k not in ["model_name", "host"]
        }

    def __infer__(self, meaning_in: str | list[dict], **kwargs: dict) -> str:
        """Infer a response from the input meaning."""
        assert isinstance(
            meaning_in, str
        ), "Currently Multimodal models are not supported. Please provide a string input."
        with open("/tmp/RawPrompt.txt", "w") as file:
            file.write(meaning_in)
        model = str(kwargs.get("model_name", self.model_name))
        if not self.check_model(model):
            self.download_model(model)
        model_params = {k: v for k, v in kwargs.items() if k not in ["model_name"]}
        messages = [{"role": "user", "content": meaning_in}]
        output = self.client.chat(
            model=model,
            messages=messages,
            options={**self.default_model_params, **model_params},
        )
        print(output)
        request_info = {}
        request_info["usage"] = {
            "prompt_tokens": output.get("prompt_eval_count"),
            "completion_tokens": output.get("eval_count"),
            "total_tokens": output.get("prompt_eval_count") + output.get("eval_count"),

        }
        response_path = "/tmp/RawResponse.json"
        requests = []
        if os.path.exists(response_path):
            with open(response_path, "r") as file:
                requests = json.load(file)
        requests.append(request_info)
        with open("/tmp/RawResponse.json", "w") as file:
            json.dump(requests, file)
        return output["message"]["content"]

    def check_model(self, model_name: str) -> bool:
        """Check if the model is available."""
        try:
            self.client.show(model_name)
            return True
        except Exception:
            return False

    def download_model(self, model_name: str) -> None:
        """Download the model."""
        self.client.pull(model_name)
