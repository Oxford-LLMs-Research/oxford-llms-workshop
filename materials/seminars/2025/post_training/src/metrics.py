import torch
import transformers
import tqdm
import asyncio
import warnings

import transformers.data.metrics.squad_metrics as squad_metrics

from datasets import load_dataset, Dataset
from pydantic import BaseModel
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, Pipeline
from typing import List, Union
from src.async_eval import StructuredOutputClient


ORACLE_SYSTEM_PROMPT = """You are a critical thinking oracle specializing in identifying logical fallacies. You will receive a statement and a proposed logical fallacy. Your task is to evaluate if the proposed logical fallacy accurately describes the error in reasoning within the statement.

You must respond exclusively in a valid JSON format that conforms to the following schema:
{
    "type": "object",
    "properties": {
        "is_correct": {
            "title": "Is Correct",
            "type": "boolean",
            "description": "A boolean value, 'true' if the logical fallacy correctly identifies the error in the statement, and 'false' otherwise."
        }
    },
    "title": "OracleAnswer",
    "required": ["is_correct"]
}

Do not include any text, markdown formatting, or explanations outside of the JSON object. Your entire output must be the JSON itself."""


class OracleAnswer(BaseModel):
    """Pydantic model for the oracle's structured response."""
    is_correct: bool


def extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except IndexError:
        return ""


class MetricCalculator:
    """
    Calculates various metrics for model predictions, including an optional AI-based oracle evaluation.
    """
    def __init__(
        self,
        metrics: List[str],
        oracle_system_prompt: str,
        oracle_model_name: str,
    ):
        supported_metrics = ["f1", "em", "oracle", "length"]
        for metric in metrics:
            if metric not in supported_metrics:
                raise ValueError(f"Metric '{metric}' is unsupported. Supported metrics are: {supported_metrics}")

        self.metrics = metrics
        self.oracle_asker = None

        if "oracle" in self.metrics:
            self.oracle_asker = StructuredOutputClient(
                response_model=OracleAnswer,
                oracle_system_prompt=oracle_system_prompt,
                oracle_model_name=oracle_model_name,
            )

        self.reset_metrics()

    def _calculate_metrics(self) -> dict:
        assert len(self.true_answers) == len(self.pred_answers) == len(self.statements)

        metrics = {metric_name: [] for metric_name in self.metrics}
        oracle_user_prompts = []

        for true_answer, pred_answer, statement in zip(self.true_answers, self.pred_answers, self.statements):
            if "em" in self.metrics:
                metrics["em"].append(squad_metrics.compute_exact(true_answer, pred_answer))
            if "f1" in self.metrics:
                metrics["f1"].append(squad_metrics.compute_f1(true_answer, pred_answer))
            if "length" in self.metrics:
                metrics["length"].append(len(pred_answer))
            if "oracle" in self.metrics:
                oracle_user_prompts.append(f'Statement: "{statement}"\nLogical Fallacy: "{pred_answer}"')

        if "oracle" in self.metrics and self.oracle_asker:
            oracle_results = asyncio.run(self.oracle_asker.run_in_parallel(user_prompts=oracle_user_prompts))

            for i, oracle_result in enumerate(oracle_results):
                if isinstance(oracle_result, OracleAnswer):
                    metrics["oracle"].append(int(oracle_result.is_correct))
                else:
                    warnings.warn(
                        f"Oracle evaluation failed for statement: '{self.statements[i]}'. "
                        "The response was invalid or could not be parsed. Scoring as incorrect."
                    )
                    metrics["oracle"].append(0)
        return metrics

    def reset_metrics(self):
        self.true_answers: List[str] = []
        self.pred_answers: List[str] = []
        self.statements: List[str] = []
    
    def update_metrics(self, true_answer: str, pred_answer: str, statement: str):
        self.true_answers.append(true_answer)
        self.pred_answers.append(pred_answer)
        self.statements.append(statement)

    def result_metrics(self) -> dict:
        metrics = self._calculate_metrics()
        
        final_results = {}
        for metric_name, values in metrics.items():
            if values:
                final_results[metric_name] = sum(values) / len(values)
            else:
                final_results[metric_name] = 0.0

        return final_results


class ModelEvaluator:
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        system_prompt: str,
    ):
        if isinstance(model, str):
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
            model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        elif isinstance(model, PreTrainedModel):
            tokenizer = AutoTokenizer.from_pretrained(model.name_or_path, use_fast=False)
            model_obj = model
        else:
            raise TypeError("The 'model' argument must be a string (model name) or a PreTrainedModel instance.")

        self.pipeline: Pipeline = transformers.pipeline(
            "text-generation",
            model=model_obj,
            tokenizer=tokenizer,
        )
        self.system_prompt = system_prompt

    def _create_prompt(self, user_prompt: str) -> List[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def eval(self,
        dataset: Union[str, Dataset],
        dataset_split: str = "test",
        metrics: List[str] = ["f1", "em"], # TODO: Fixed with List[str] type hint
        num_samples: int = -1,
        batch_size: int = 8,
        parse_output: bool = False,
        oracle_system_prompt: str = ORACLE_SYSTEM_PROMPT,
        oracle_model_name: str = "openai/gpt-oss-120b",
    ) -> dict:
        if isinstance(dataset, str):
            dataset = load_dataset(dataset, split=dataset_split)

        if num_samples > 0:
            min_number = min(len(dataset), num_samples)
            dataset = dataset.select(range(min_number))

        metric_calculator = MetricCalculator(
            metrics,
            oracle_system_prompt=oracle_system_prompt,
            oracle_model_name=oracle_model_name,
        )

        for i in tqdm.tqdm(range(0, len(dataset), batch_size), desc="Model eval"):
            batch = dataset[i : i + batch_size]
            batch_articles = batch['source_article']
            batch_prompts = [self._create_prompt(article) for article in batch_articles]
            
            outputs = self.pipeline(
                batch_prompts,
                do_sample=False,
                return_full_text=False,
            )

            pred_answers = [ans[0]['generated_text'] for ans in outputs]
            true_answers = batch['logical_fallacies']
            
            for true_answer, pred_answer, article in zip(true_answers, pred_answers, batch_articles):
                if parse_output:
                    pred_answer = extract_xml_answer(pred_answer)
                metric_calculator.update_metrics(true_answer, pred_answer, article)

        return metric_calculator.result_metrics()
