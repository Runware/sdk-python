import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IPromptEnhancer, IEnhancedPrompt

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    prompt = "A beautiful sunset over the mountains"
    print(f"Original Prompt: {prompt}")

    prompt_enhancer = IPromptEnhancer(
        prompt=prompt,
        prompt_versions=3,
    )
    try:
        enhanced_prompts: List[IEnhancedPrompt] = await runware.enhancePrompt(
            promptEnhancer=prompt_enhancer
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Enhanced Prompts:\n")
    for enhanced_prompt in enhanced_prompts:
        print(enhanced_prompt.text, "\n")


if __name__ == "__main__":
    asyncio.run(main())
