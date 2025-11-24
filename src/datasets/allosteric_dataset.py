import logging

from .binding_dataset import BindingDataModule

log = logging.getLogger(__name__)


class AllostericDataModule(BindingDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def test_dataloader_indices(self) -> dict[str, int]:
        return {
            "allosteric": 0
        }

    def test_dataloader(self):
        return [
            self._create_dataloader("allosteric")
        ]
