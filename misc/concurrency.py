import os
import httpx
import time
import asyncio
import requests
from threading import current_thread, Thread
from multiprocessing import cpu_count, Pool, current_process
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait

# testing parameters
NUM_IO_TASKS = 100
NUM_PRIME_START = 1000
NUM_PRIME_END = 16000
MAX_WORKERS = cpu_count() - 1
URL = 'https://httpbin.org/ip'


def run_tests():
    # test various techniques to speed up IO/CPU bound operations
    tests = [
        ('IO Bound Sync', io_bound_sync),
        ('IO Bound Threading', io_bound_threading),
        ('IO Bound Thread Pool', io_bound_thread_pool),
        ('IO Bound Async', io_bound_async),
        ('CPU Bound Sync', cpu_bound_sync),
        ('CPU Bound Multiprocessing Pool', cpu_bound_multiprocessing_pool),
        ('CPU Bound Process Pool Executor', cpu_bound_process_pool_executor),
    ]

    results = []
    for name, test in tests:
        start_time = time.perf_counter()
        test()
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        results.append((name, elapsed_time))

    print(f"{'Test Name':<35} {'Elapsed Time(seconds)':<20}")
    print('-' * 55)
    for name, elapsed_time in results:
        print(f'{name:<35} {elapsed_time:<20.4f}')


def make_request(num):
    # make a synchronous HTTP request
    pid = os.getpid()
    thread_name = current_thread().name
    process_name = current_process().name
    print(f'{pid} - {process_name} - {thread_name}')
    requests.get(URL)


async def make_request_async(num, client):
    # make an asynchronous HTTP request
    pid = os.getpid()
    thread_name = current_thread().name
    process_name = current_process().name
    print(f'{pid} - {process_name} - {thread_name}')
    await client.get(URL)


def get_prime_numbers(num):
    # calculate prime numbers up to a given number
    pid = os.getpid()
    thread_name = current_thread().name
    process_name = current_process().name
    print(f'{pid} - {process_name} - {thread_name}')
    numbers = []
    prime = [True for _ in range(num + 1)]
    p = 2
    while p * p <= num:
        if prime[p]:
            for i in range(p * 2, num + 1, p):
                prime[i] = False
        p += 1
    prime[0] = prime[1] = False
    for p in range(num + 1):
        if prime[p]:
            numbers.append(p)
    return numbers


def io_bound_sync():
    # perform synchronous IO-bound tasks
    for num in range(1, NUM_IO_TASKS + 1):
        make_request(num)


def io_bound_threading():
    # perform IO-bound tasks using threading
    tasks = []
    for num in range(1, NUM_IO_TASKS + 1):
        task = Thread(target=make_request, args=(num,))
        tasks.append(task)
        task.start()
    for task in tasks:
        task.join()


def io_bound_thread_pool():
    # perform IO-bound tasks using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_IO_TASKS) as executor:
        futures = [executor.submit(make_request, num) for num in range(1, NUM_IO_TASKS + 1)]
    wait(futures)


def io_bound_async():
    # perform asynchronous IO-bound tasks
    async def _concurrent_async():
        async with httpx.AsyncClient() as client:
            await asyncio.gather(*[make_request_async(num, client) for num in range(1, NUM_IO_TASKS + 1)])
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_concurrent_async())


def cpu_bound_sync():
    # perform synchronous CPU-bound tasks
    for num in range(NUM_PRIME_START, NUM_PRIME_END):
        get_prime_numbers(num)


def cpu_bound_multiprocessing_pool():
    # perform CPU-bound tasks using multiprocessing Pool
    with Pool(MAX_WORKERS) as p:
        p.starmap(get_prime_numbers, zip(range(NUM_PRIME_START, NUM_PRIME_END)))
        p.close()
        p.join()


def cpu_bound_process_pool_executor():
    # perform CPU-bound tasks using ProcessPoolExecutor
    with ProcessPoolExecutor(MAX_WORKERS) as executor:
        futures = [executor.submit(get_prime_numbers, num) for num in range(NUM_PRIME_START, NUM_PRIME_END)]
    wait(futures)


if __name__ == '__main__':
    run_tests()
