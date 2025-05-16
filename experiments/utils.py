import asyncio
import functools

def retry(times, failed_return, exceptions, backoff_factor=1):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < times:
                try:
                    return await func(*args, **kwargs, retry_attempt=attempt)
                except exceptions as e:
                    print(
                        f"Exception [{type(e)}:{e}] thrown when attempting to run {func}, attempt {attempt} of {times}"
                    )
                    await asyncio.sleep(backoff_factor * 2**attempt)
                    attempt += 1
            return failed_return
        return wrapper
    return decorator
