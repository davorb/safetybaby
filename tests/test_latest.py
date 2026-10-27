import asyncio

from babymon.latest import Latest


def test_empty_cell_reads_none():
    assert Latest[int]().get() is None


async def test_reader_gets_the_newest_value_and_version():
    cell: Latest[str] = Latest()
    cell.set("a")
    cell.set("b")
    assert cell.get() == "b"
    assert await cell.newer_than(0) == (2, "b")


async def test_reader_waits_for_a_newer_version():
    cell: Latest[str] = Latest()
    cell.set("a")
    waiter = asyncio.create_task(cell.newer_than(1))
    await asyncio.sleep(0)
    assert not waiter.done()
    cell.set("b")
    assert await waiter == (2, "b")


async def test_slow_reader_skips_intermediate_values():
    cell: Latest[int] = Latest()
    version = 0
    received = []
    for i in range(3):
        cell.set(i)
    version, value = await cell.newer_than(version)
    received.append(value)
    assert received == [2]


async def test_every_waiter_wakes_on_one_set():
    cell: Latest[str] = Latest()
    waiters = [asyncio.create_task(cell.newer_than(0)) for _ in range(3)]
    await asyncio.sleep(0)
    cell.set("x")
    assert await asyncio.gather(*waiters) == [(1, "x")] * 3
