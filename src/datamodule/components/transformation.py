from typing import Any
import hydra
from omegaconf import DictConfig
from torchvision.transforms import transforms as TT



class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        augmentations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
            augmentation = hydra.utils.instantiate(
                transforms_cfg.get(augmentation_name), _convert_="object"
            )
            augmentations.append(augmentation)
        self.augmentations = TT.Compose(augmentations)


    def __call__(self, image: Any, **kwargs: Any) -> Any:
        """Apply TransformsWrapper module.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """
        return self.augmentations(image, **kwargs)


