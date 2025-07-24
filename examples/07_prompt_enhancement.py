"""
Prompt Enhancement Example

This example demonstrates how to use Runware's prompt enhancement feature
to automatically improve and expand prompts for better image generation results.
The enhanced prompts include more descriptive language, artistic terms, and
technical specifications that lead to higher quality outputs.
"""

import asyncio
import os
from typing import List

from runware import (
    IEnhancedPrompt,
    IImage,
    IImageInference,
    IPromptEnhance,
    Runware,
    RunwareError,
)


async def basic_prompt_enhancement():
    """Enhance simple prompts to create more detailed descriptions."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Simple prompts to enhance
        simple_prompts = [
            "a cat",
            "sunset over mountains",
            "beautiful woman",
            "futuristic city",
        ]

        for original_prompt in simple_prompts:
            print(f"\nOriginal prompt: '{original_prompt}'")

            enhancer = IPromptEnhance(
                prompt=original_prompt,
                promptVersions=3,  # Generate 3 different enhanced versions
                promptMaxLength=200,  # Maximum length for enhanced prompts
                includeCost=True,
            )

            enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
                promptEnhancer=enhancer
            )

            print("Enhanced versions:")
            for i, enhanced in enumerate(enhanced_prompts, 1):
                print(f"  {i}. {enhanced.text}")
                if enhanced.cost:
                    print(f"     Cost: ${enhanced.cost}")

    except RunwareError as e:
        print(f"Error in basic prompt enhancement: {e}")
    finally:
        await runware.disconnect()


async def artistic_prompt_enhancement():
    """Enhance prompts specifically for artistic and creative image generation."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Artistic prompts that can benefit from enhancement
        artistic_prompts = [
            "abstract painting",
            "portrait in renaissance style",
            "cyberpunk street scene",
            "magical forest",
        ]

        for prompt in artistic_prompts:
            print(f"\nArtistic prompt: '{prompt}'")

            enhancer = IPromptEnhance(
                prompt=prompt,
                promptVersions=2,
                promptMaxLength=300,  # Longer prompts for artistic detail
                includeCost=True,
            )

            enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
                promptEnhancer=enhancer
            )

            print("Enhanced artistic descriptions:")
            for i, enhanced in enumerate(enhanced_prompts, 1):
                print(f"  Version {i}:")
                print(f"    {enhanced.text}")
                if enhanced.cost:
                    print(f"    Processing cost: ${enhanced.cost}")

    except RunwareError as e:
        print(f"Error in artistic prompt enhancement: {e}")
    finally:
        await runware.disconnect()


async def photography_prompt_enhancement():
    """Enhance prompts for photorealistic image generation with technical terms."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Photography-focused prompts
        photography_prompts = [
            "professional headshot",
            "landscape photography",
            "macro flower photo",
            "street photography at night",
        ]

        for prompt in photography_prompts:
            print(f"\nPhotography prompt: '{prompt}'")

            enhancer = IPromptEnhance(
                prompt=prompt, promptVersions=2, promptMaxLength=250, includeCost=True
            )

            enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
                promptEnhancer=enhancer
            )

            print("Enhanced with photography terms:")
            for i, enhanced in enumerate(enhanced_prompts, 1):
                print(f"  Version {i}: {enhanced.text}")

    except RunwareError as e:
        print(f"Error in photography prompt enhancement: {e}")
    finally:
        await runware.disconnect()


async def compare_original_vs_enhanced():
    """Compare image generation results using original vs enhanced prompts."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        original_prompt = "a dragon in a castle"
        print(f"Comparing results for: '{original_prompt}'")

        # First, enhance the prompt
        enhancer = IPromptEnhance(
            prompt=original_prompt,
            promptVersions=1,  # Just one enhanced version for comparison
            promptMaxLength=180,
        )

        enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
            promptEnhancer=enhancer
        )

        enhanced_prompt = enhanced_prompts[0].text
        print(f"Enhanced to: '{enhanced_prompt}'")

        # Generate image with original prompt
        print("\nGenerating with original prompt...")
        original_request = IImageInference(
            positivePrompt=original_prompt,
            model="civitai:4384@128713",
            numberResults=1,
            height=768,
            width=768,
            seed=42,  # Fixed seed for fair comparison
        )

        original_images: List[IImage] = await runware.imageInference(
            requestImage=original_request
        )

        # Generate image with enhanced prompt
        print("Generating with enhanced prompt...")
        enhanced_request = IImageInference(
            positivePrompt=enhanced_prompt,
            model="civitai:4384@128713",
            numberResults=1,
            height=768,
            width=768,
            seed=42,  # Same seed for comparison
        )

        enhanced_images: List[IImage] = await runware.imageInference(
            requestImage=enhanced_request
        )

        # Display results
        print("\nComparison Results:")
        print(f"Original prompt result: {original_images[0].imageURL}")
        print(f"Enhanced prompt result: {enhanced_images[0].imageURL}")
        print(
            "\nThe enhanced prompt should produce more detailed and visually appealing results!"
        )

    except RunwareError as e:
        print(f"Error in comparison: {e}")
    finally:
        await runware.disconnect()


async def batch_prompt_enhancement():
    """Enhance multiple prompts efficiently in batch."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Batch of prompts to enhance
        prompt_batch = [
            "cozy coffee shop",
            "space exploration",
            "underwater scene",
            "medieval knight",
            "tropical paradise",
        ]

        print("Batch enhancing multiple prompts...")

        # Create enhancement tasks
        enhancement_tasks = []
        for prompt in prompt_batch:
            enhancer = IPromptEnhance(
                prompt=prompt, promptVersions=1, promptMaxLength=150
            )
            task = runware.promptEnhance(promptEnhancer=enhancer)
            enhancement_tasks.append((prompt, task))

        # Execute all enhancements concurrently
        results = await asyncio.gather(
            *[task for _, task in enhancement_tasks], return_exceptions=True
        )

        # Process results
        print("\nBatch Enhancement Results:")
        for i, (original_prompt, result) in enumerate(zip(prompt_batch, results)):
            print(f"\n{i + 1}. Original: '{original_prompt}'")

            if isinstance(result, Exception):
                print(f"   Error: {result}")
            else:
                enhanced_prompts: List[IEnhancedPrompt] = result
                if enhanced_prompts:
                    print(f"   Enhanced: '{enhanced_prompts[0].text}'")

    except RunwareError as e:
        print(f"Error in batch enhancement: {e}")
    finally:
        await runware.disconnect()


async def length_variation_testing():
    """Test prompt enhancement with different maximum length settings."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        base_prompt = "magical wizard casting spells"
        lengths = [50, 100, 200, 300]

        print(f"Testing different enhancement lengths for: '{base_prompt}'")

        for max_length in lengths:
            print(f"\nMax length: {max_length} characters")

            enhancer = IPromptEnhance(
                prompt=base_prompt, promptVersions=1, promptMaxLength=max_length
            )

            enhanced_prompts: List[IEnhancedPrompt] = await runware.promptEnhance(
                promptEnhancer=enhancer
            )

            if enhanced_prompts:
                enhanced_text = enhanced_prompts[0].text
                actual_length = len(enhanced_text)
                print(f"  Result ({actual_length} chars): {enhanced_text}")

    except RunwareError as e:
        print(f"Error in length variation testing: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all prompt enhancement examples."""
    print("=== Basic Prompt Enhancement ===")
    await basic_prompt_enhancement()

    print("\n=== Artistic Prompt Enhancement ===")
    await artistic_prompt_enhancement()

    print("\n=== Photography Prompt Enhancement ===")
    await photography_prompt_enhancement()

    print("\n=== Original vs Enhanced Comparison ===")
    await compare_original_vs_enhanced()

    print("\n=== Batch Prompt Enhancement ===")
    await batch_prompt_enhancement()

    print("\n=== Length Variation Testing ===")
    await length_variation_testing()


if __name__ == "__main__":
    asyncio.run(main())
