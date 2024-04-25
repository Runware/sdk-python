import asyncio
from runware import getIntervalWithPromise


async def simulate_api_call(task_uuid):
    print(f"Simulating API call for task {task_uuid}")
    await asyncio.sleep(2)  # Simulating an asynchronous API call
    print(f"API call completed for task {task_uuid}")
    global result
    result = f"Result for task {task_uuid}"


async def main():
    # Test case 1: Callback resolves before timeout
    print("Test Case 1: Callback resolves before timeout")
    task1_uuid = "task1"

    # Start the API call simulation
    asyncio.create_task(simulate_api_call(task1_uuid))

    result1 = await getIntervalWithPromise(
        lambda params: params["resolve"](result) if "result" in globals() else None,
        debugKey="api_call_1",
        timeOutDuration=5000,  # 5 seconds
        pollingInterval=500,  # 0.5 seconds
    )
    print(f"Result 1: {result1}")

    # Test case 2: Callback rejects before timeout
    print("\nTest Case 2: Callback rejects before timeout")
    task2_uuid = "task2"
    try:
        await getIntervalWithPromise(
            lambda params: (
                params["reject"]("API call failed")
                if not "result" in globals()
                else None
            ),
            debugKey="api_call_2",
            timeOutDuration=5000,  # 5 seconds
            pollingInterval=500,  # 0.5 seconds
        )
    except Exception as e:
        print(f"Error: {str(e)}")

    # Test case 3: Timeout occurs before callback resolves
    print("\nTest Case 3: Timeout occurs before callback resolves")
    task3_uuid = "task3"
    try:
        await getIntervalWithPromise(
            lambda params: None,
            debugKey="api_call_3",
            timeOutDuration=3000,  # 3 seconds
            pollingInterval=500,  # 0.5 seconds
        )
    except Exception as e:
        print(f"Error: {str(e)}")

    # Test case 4: Callback resolves with custom polling interval
    print("\nTest Case 4: Callback resolves with custom polling interval")
    task4_uuid = "task4"
    result4 = await getIntervalWithPromise(
        lambda params: params["resolve"](result) if "result" in globals() else None,
        debugKey="api_call_4",
        timeOutDuration=10000,  # 10 seconds
        pollingInterval=1000,  # 1 second
    )
    print(f"Result 4: {result4}")


if __name__ == "__main__":
    asyncio.run(main())
