"""
Model Management Example

This example demonstrates model management capabilities including:
- Searching for models in the repository
- Uploading custom models (checkpoints, LoRA, ControlNet)
- Managing model metadata and configurations
- Working with different model architectures
"""

import asyncio
import os

from runware import (
    EModelArchitecture,
    IModelSearch,
    IModelSearchResponse,
    IUploadModelCheckPoint,
    IUploadModelControlNet,
    IUploadModelLora,
    IUploadModelResponse,
    Runware,
    RunwareError,
)


async def search_models_basic():
    """Search for models using basic criteria."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Basic model search
        search_request = IModelSearch(
            search="fantasy art",  # Search term
            limit=5,  # Limit results
            offset=0,  # Starting position
        )

        results: IModelSearchResponse = await runware.modelSearch(
            payload=search_request
        )

        print("Basic Model Search Results:")
        print(f"Total models found: {results.totalResults}")
        print(f"Showing first {len(results.results)} models:")

        for i, model in enumerate(results.results, 1):
            print(f"\n{i}. {model.name} (v{model.version})")
            print(f"   Category: {model.category}")
            print(f"   Architecture: {model.architecture}")
            print(f"   AIR: {model.air}")
            print(f"   Tags: {', '.join(model.tags) if model.tags else 'None'}")
            if model.comment:
                print(f"   Description: {model.comment}")

            # Access additional fields that may have been provided by API
            if hasattr(model, "downloadURL"):
                print(f"   Download URL: {model.downloadURL}")
            if hasattr(model, "shortDescription"):
                print(f"   Short Description: {model.shortDescription[:100]}...")

    except RunwareError as e:
        print(f"Error in basic model search: {e}")
    finally:
        await runware.disconnect()


async def search_models_filtered():
    """Search for models with specific filters and categories."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Search for SDXL checkpoints
        print("Searching for SDXL checkpoint models...")
        checkpoint_search = IModelSearch(
            category="checkpoint",
            architecture=EModelArchitecture.SDXL,
            visibility="public",
            limit=3,
        )

        checkpoint_results: IModelSearchResponse = await runware.modelSearch(
            payload=checkpoint_search
        )

        print(f"Found {checkpoint_results.totalResults} SDXL checkpoints:")
        for model in checkpoint_results.results:
            print(f"  - {model.name}: {model.air}")
            if model.defaultSteps:
                print(f"    Default steps: {model.defaultSteps}")
            if model.defaultCFG:
                print(f"    Default CFG: {model.defaultCFG}")

        # Search for LoRA models
        print("\nSearching for LoRA models...")
        lora_search = IModelSearch(category="lora", tags=["anime", "style"], limit=3)

        lora_results: IModelSearchResponse = await runware.modelSearch(
            payload=lora_search
        )

        print(f"Found {lora_results.totalResults} LoRA models:")
        for model in lora_results.results:
            print(f"  - {model.name}: {model.air}")
            if hasattr(model, "defaultWeight") and model.defaultWeight:
                print(f"    Default weight: {model.defaultWeight}")
            if model.positiveTriggerWords:
                print(f"    Trigger words: {model.positiveTriggerWords}")

        # Search for ControlNet models
        print("\nSearching for ControlNet models...")
        controlnet_search = IModelSearch(
            category="controlnet", architecture=EModelArchitecture.SDXL, limit=3
        )

        controlnet_results: IModelSearchResponse = await runware.modelSearch(
            payload=controlnet_search
        )

        print(f"Found {controlnet_results.totalResults} ControlNet models:")
        for model in controlnet_results.results:
            print(f"  - {model.name}: {model.air}")
            if hasattr(model, "conditioning") and model.conditioning:
                print(f"    Conditioning: {model.conditioning}")

    except RunwareError as e:
        print(f"Error in filtered model search: {e}")
    finally:
        await runware.disconnect()


async def upload_checkpoint_model():
    """Upload a custom checkpoint model."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure checkpoint model upload
        checkpoint_payload = IUploadModelCheckPoint(
            air="runware:68487@0862923414",  # Unique AIR identifier
            name="SDXL Model",
            heroImageURL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/01.png?download=true",
            downloadURL="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors?download=true",
            uniqueIdentifier="unique_checkpoint_id_12345678901234567890123456789012",
            version="1.0",
            tags=["realistic", "portrait", "photography"],
            architecture="sdxl",
            type="base",  # Required for checkpoints
            defaultWeight=1.0,
            format="safetensors",
            positiveTriggerWords="masterpiece, best quality",
            shortDescription="High-quality realistic portrait model",
            private=False,
            defaultScheduler="DPM++ 2M Karras",  # Required for checkpoints
            defaultSteps=30,
            defaultGuidanceScale=7.5,
            comment="Custom trained model for realistic portraits",
        )

        print("Uploading checkpoint model...")
        upload_result: IUploadModelResponse = await runware.modelUpload(
            requestModel=checkpoint_payload
        )

        if upload_result:
            print(f"Checkpoint upload successful!")
            print(f"  AIR: {upload_result.air}")
            print(f"  Task UUID: {upload_result.taskUUID}")
            print(f"  Task Type: {upload_result.taskType}")
        else:
            print("Checkpoint upload failed")

    except RunwareError as e:
        print(f"Error uploading checkpoint model: {e}")
    finally:
        await runware.disconnect()


async def upload_lora_model():
    """Upload a custom LoRA model."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure LoRA model upload
        lora_payload = IUploadModelLora(
            air="runware:68487@08629",
            name="Ghibli lora",
            heroImageURL="https://huggingface.co/ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style/resolve/main/images/Studio%20Ghibli%20style_17_3.0.png",
            downloadURL="https://huggingface.co/ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style/resolve/main/Studio%20Ghibli%20style.safetensors?download=true",
            uniqueIdentifier="unique_lora_id_abcdefghijklmnopqrstuvwxyz123456789012",
            version="2.0",
            tags=["anime", "style", "cartoon"],
            architecture="sdxl",
            format="safetensors",
            defaultWeight=0.8,  # Typical LoRA weight
            positiveTriggerWords="anime style, cel shading",
            shortDescription="Anime art style enhancement LoRA",
            private=False,
            comment="Trained on high-quality anime artwork",
        )

        print("Uploading LoRA model...")
        upload_result: IUploadModelResponse = await runware.modelUpload(
            requestModel=lora_payload
        )

        if upload_result:
            print(f"LoRA upload successful!")
            print(f"  AIR: {upload_result.air}")
            print(f"  Task UUID: {upload_result.taskUUID}")
        else:
            print("LoRA upload failed")

    except RunwareError as e:
        print(f"Error uploading LoRA model: {e}")
    finally:
        await runware.disconnect()


async def upload_controlnet_model():
    """Upload a custom ControlNet model."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure ControlNet model upload
        controlnet_payload = IUploadModelControlNet(
            air="runware:68487@08629112",
            name="Custom Canny ControlNet",
            downloadURL="https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.safetensors?download=true",
            uniqueIdentifier="unique_controlnet_id_987654321098765432109876543210",
            version="1.5",
            tags=["controlnet", "canny", "edge"],
            architecture="sdxl",
            format="safetensors",
            conditioning="canny",  # Required for ControlNet
            shortDescription="Canny edge detection ControlNet for SDXL",
            private=False,
            comment="Fine-tuned for better edge detection accuracy",
        )

        print("Uploading ControlNet model...")
        upload_result: IUploadModelResponse = await runware.modelUpload(
            requestModel=controlnet_payload
        )

        if upload_result:
            print(f"ControlNet upload successful!")
            print(f"  AIR: {upload_result.air}")
            print(f"  Task UUID: {upload_result.taskUUID}")
        else:
            print("ControlNet upload failed")

    except RunwareError as e:
        print(f"Error uploading ControlNet model: {e}")
    finally:
        await runware.disconnect()


async def search_runware_models():
    """Search for runware's private models."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Search for runware's private models
        private_search = IModelSearch(
            visibility="private", limit=10  # Only private models
        )

        results: IModelSearchResponse = await runware.modelSearch(
            payload=private_search
        )

        print("runware's Private Models:")
        print(f"Total private models: {results.totalResults}")

        if results.results:
            for i, model in enumerate(results.results, 1):
                print(f"\n{i}. {model.name} (v{model.version})")
                print(f"   AIR: {model.air}")
                print(f"   Category: {model.category}")
                print(f"   Private: {model.private}")
                if model.comment:
                    print(f"   Notes: {model.comment}")
        else:
            print("No private models found")

    except RunwareError as e:
        print(f"Error searching runware models: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all model management examples."""
    print("=== Basic Model Search ===")
    await search_models_basic()

    print("\n=== Filtered Model Search ===")
    await search_models_filtered()

    print("\n=== Upload Checkpoint Model ===")
    await upload_checkpoint_model()

    print("\n=== Upload LoRA Model ===")
    await upload_lora_model()

    print("\n=== Upload ControlNet Model ===")
    await upload_controlnet_model()

    print("\n=== Search runware Models ===")
    await search_runware_models()


if __name__ == "__main__":
    asyncio.run(main())
