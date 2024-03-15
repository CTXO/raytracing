from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from structures import Ray

class IntersectableMixin(ABC):
    @abstractmethod
    def intersect(self, ray: Ray) -> dict:
        raise NotImplementedError

