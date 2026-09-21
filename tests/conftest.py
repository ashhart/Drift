import torch
import pytest


def pytest_sessionstart(session):
    torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)
