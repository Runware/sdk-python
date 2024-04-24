import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IPromptEnhancer, IEnhancedPrompt

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)

    prompt = "A beautiful sunset over the mountains"

    prompt_enhancer = IPromptEnhancer(
        prompt=prompt,
        prompt_versions=3,
    )

    enhanced_prompts: List[IEnhancedPrompt] = await runware.enhancePrompt(
        promptEnhancer=prompt_enhancer
    )

    print("Enhanced Prompts:")
    for enhanced_prompt in enhanced_prompts:
        print(enhanced_prompt.text)


if __name__ == "__main__":
    asyncio.run(main())
