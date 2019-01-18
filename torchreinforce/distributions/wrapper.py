from .base import ReinforceDistribution

def getNonDeterministicWrapper(baseClass):
    class NonDeterministicWrapper(baseClass, ReinforceDistribution):
        def __init__(self, *args, **kwargs):
            if "deterministic" in kwargs: del kwargs["deterministic"]
            super(NonDeterministicWrapper, self).__init__(*args, **kwargs)
    return NonDeterministicWrapper