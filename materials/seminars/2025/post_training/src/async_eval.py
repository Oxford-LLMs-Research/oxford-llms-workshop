
import asyncio
import tqdm
import os
import json

from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI
from typing import Type, TypeVar, Generic, Optional


T = TypeVar('T', bound=BaseModel)


def get_event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


class StructuredOutputClient(Generic[T]):
    def __init__(self, 
        response_model: Type[T],
        oracle_system_prompt: str,
        oracle_model_name: str = "openai/gpt-oss-120b",
        sampling: bool = False,
    ):
        self.client = AsyncOpenAI(
            base_url=os.environ['ORACLE_BASE_URL'],
            api_key=os.environ['ORACLE_API_KEY'],
        )
        self.response_model = response_model
        self.oracle_system_prompt = oracle_system_prompt
        self.model = oracle_model_name
        self.temperature = 0.0 if sampling else None

    async def get_oracle_answer_async(self, user_prompt: str) -> Optional[T]:
        messages = [
            {"role": "system", "content": self.oracle_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        guided_json_schema = self.response_model.model_json_schema()

        completions = await self.client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            extra_body={"guided_json": guided_json_schema},
            temperature=self.temperature,
            timeout=30.0,
        )

        try:
            content_dict = json.loads(completions.choices[0].message.content)
            content_dict['reasoning'] = completions.choices[0].message.reasoning_content
            oracle_result = self.response_model.model_validate(content_dict)
        except (IndexError, json.JSONDecodeError, ValidationError) as e:
            print(f"An error occurred while parsing the response: {e}")
            oracle_result = None

        return oracle_result

    def get_oracle_answer(self, user_prompt: str) -> Optional[T]:
        loop = get_event_loop()
        return loop.run_until_complete(
            self.get_oracle_answer_async(
                user_prompt=user_prompt,
            )
        )

    async def run_in_parallel(
        self,
        user_prompts: list[str],
        concurrency_limit: int = 50,
    ):
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def get_answer_with_semaphore(prompt: str) -> Optional[T]:
            async with semaphore:
                return await self.get_oracle_answer_async(prompt)

        tasks = [get_answer_with_semaphore(prompt) for prompt in user_prompts]
        all_results = await tqdm.asyncio.tqdm.gather(*tasks, desc="Running Oracle Evaluation")
        return all_results
