from abc import ABC, abstractmethod
from typing import Sequence
import time
import pyRAPL


class _Measurement(ABC):
    measurement_results: Sequence[any]

    @abstractmethod
    def setup(self):
        """Setup the measurement environment."""
        pass

    @abstractmethod
    def begin(self):
        """Start the measurement."""
        pass

    @abstractmethod
    def end(self):
        """End the measurement and return the result."""
        pass

    @abstractmethod
    def results(self) -> Sequence[any]:
        """Get the result of the measurement."""
        pass


class TimeMeasurement(_Measurement):
    start_time: float

    def __init__(self):
        self.start_time = 0
        self.measurement_results = []

    def setup(self):
        self.measurement_results = []

    def begin(self):
        self.start_time = time.process_time_ns()

    def end(self):
        end_time = time.process_time_ns()
        elapsed_time = end_time - self.start_time
        self.measurement_results.append(elapsed_time)

    def results(self) -> Sequence[float]:
        return self.measurement_results


class EnergyMeasurement(_Measurement):
    meter: pyRAPL.Measurement

    def __init__(self):
        self.measurement_results = []
        self.meter = None

    def setup(self):
        self.measurement_results = []
        pyRAPL.setup()
        self.meter = pyRAPL.Measurement("nmlr_experiment")

    def begin(self):
        self.meter.begin()

    def end(self):
        self.meter.end()
        self.measurement_results.append(self.meter.result)

    def results(self) -> Sequence[pyRAPL.Result]:
        return self.measurement_results
