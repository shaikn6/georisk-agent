from georisk.providers.base import ProviderError
from georisk.providers.chain import ProviderChain


class P:
    def __init__(self, name, ok=True):
        self.name, self.ok = name, ok

    def probe(self):
        if not self.ok:
            raise ProviderError(f"{self.name} unreachable: dns")


def test_health_reports_each_provider_independently():
    chain = ProviderChain([P("a"), P("b", ok=False)], sleep=lambda s: None)
    h = chain.health()
    assert h["a"] == {"reachable": True, "circuit": "closed"}
    assert h["b"]["reachable"] is False and "unreachable" in h["b"]["error"] and h["b"]["circuit"] == "closed"
