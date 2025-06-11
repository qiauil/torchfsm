
class NanSimulationError(RuntimeError):
    """Raised when a simulation produces NaN values."""
    def __init__(self, message="Simulation produced NaN values."):
        super().__init__(message)
        
        
class OutOfMemoryError(RuntimeError):
    """Raised when a simulation runs out of memory."""
    def __init__(self, message="Simulation ran out of memory."):
        super().__init__(message)