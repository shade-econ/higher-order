"""Reuse the paper steady state and derivative operators across tests."""

import pytest

from smooth_sim.model import paper_household
from analytical_simple.backward import Backward as SimpleBackward
from analytical_simple.forward import Forward as SimpleForward
from analytical_fast.backward import Backward as FastBackward
from analytical_fast.forward import Forward as FastForward


@pytest.fixture(scope="session")
def household():
    return paper_household()


@pytest.fixture(scope="session")
def simple_derivatives(household):
    hh, ss = household
    return SimpleBackward(hh, ss), SimpleForward(hh, ss)


@pytest.fixture(scope="session")
def fast_derivatives(household):
    hh, ss = household
    return FastBackward(hh, ss), FastForward(hh, ss)
