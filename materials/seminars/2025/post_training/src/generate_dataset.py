import asyncio
import os
import tqdm
import json

from src.async_eval import StructuredOutputClient, get_event_loop, T
from datasets import Dataset
from typing import Type, Callable

"""
def row_preprocessor(data_row: dict[str, str]) -> str:
  pass

def row_postprocessor(
  response: StructuredReasoningResponse,
  system_prompt: str,
  data_row: dict[str, str],
) -> dict[str, str]:
  pass
  
"""

async def create_dataset_row(
    client: StructuredOutputClient[T], 
    row_preprocessor: Callable,
    row_postprocessor: Callable,
    data_row: dict, 
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        user_prompt = row_preprocessor(data_row)

        # Asynchronously get the structured response
        response = await client.get_oracle_answer_async(user_prompt)
        
        # Return the formatted dictionary
        return row_postprocessor(response, system_prompt, data_row)


async def generate_dataset_async(
    client: StructuredOutputClient[T],
    row_preprocessor: Callable,
    row_postprocessor: Callable,
    source_data: list[dict], 
    system_prompt: str,
    concurrency_limit: int = 50,
) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [
        create_dataset_row(
            client,
            row_preprocessor,
            row_postprocessor,
            data_row,
            system_prompt,
            semaphore,
        ) for data_row in source_data
    ]
    results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Creating dataset")
    successful_results = [res for res in results if (not isinstance(res, Exception) and not res is None)]

    # Log any errors that occurred
    errors = [res for res in results if isinstance(res, Exception)]
    if errors:
        print(f"\nEncountered {len(errors)} errors. First error: {errors[0]}")

    return successful_results


def generate_preprocessed_dataset(
    original_dataset: Dataset,
    response_model: Type[T],
    row_preprocessor: Callable,
    row_postprocessor: Callable,
    oracle_system_prompt: str,
    dataset_system_prompt: str,
    result_dataset_name: str,
) -> None:
    oracle_evaluator = StructuredOutputClient(
        response_model=response_model,
        oracle_system_prompt=oracle_system_prompt,
        sampling=True,
    )

    loop = get_event_loop()
    final_dataset = loop.run_until_complete(
        generate_dataset_async(
            client=oracle_evaluator,
            row_preprocessor=row_preprocessor,
            row_postprocessor=row_postprocessor,
            source_data=original_dataset,
            system_prompt=dataset_system_prompt,
        )
    )
    if len(final_dataset) == 0:
        print("Dataset is empty. Skipping writing")
        return

    dataset_name = os.path.join("datasets", result_dataset_name)
    print(f"Writing result dataset to {dataset_name}")
    with open(dataset_name, 'w') as f:
        for entry in final_dataset:
            json.dump(entry, f)
            f.write('\n')
