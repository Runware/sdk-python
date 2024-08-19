import asyncio
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IPromptEnhance, IEnhancedPrompt, RunwareAPIError

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    prompt = "A beautiful sunset over the mountains"
    print(f"Original Prompt: {prompt}")

    # With only mandatory parameters
    prompt_enhancer = IPromptEnhance(
        prompt=prompt,
        promptVersions=3,
        promptMaxLength=64,
    )
    # With all parameters
    prompt_enhancer = IPromptEnhance(
        prompt=prompt,
        promptVersions=3,
        promptMaxLength=300,
        includeCost=True,
    )
    try:
        enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
            promptEnhancer=prompt_enhancer
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Enhanced Prompts:\n")
        for enhanced_prompt in enhanced_prompts:
            print(enhanced_prompt.text, "\n")
            # print(enhanced_prompt.cost, "\n")


if __name__ == "__main__":
    asyncio.run(main())
