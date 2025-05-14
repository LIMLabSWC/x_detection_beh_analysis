from detectron2.engine.defaults import DefaultPredictor
import torch
from torch.multiprocessing import Pool, cpu_count

class BatchPredictor(DefaultPredictor):
    """
    A predictor that supports batch prediction for multiple images.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    """

    def __init__(self, cfg, use_multiprocessing=False):
        super().__init__(cfg)
        self.use_multiprocessing = use_multiprocessing

    @torch.jit.script
    def _process_image(self, original_image: torch.Tensor, input_format: str, aug, device: str):
        """
        Preprocess a single image using TorchScript for optimization.
        """
        if input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        image = image.to(device)
        return {"image": image, "height": height, "width": width}

    def predict_batch(self, images):
        """
        Predict a batch of images.

        Args:
            images (list[np.ndarray]): A list of images, each of shape (H, W, C) (in BGR order).

        Returns:
            list[dict]: A list of predictions, one for each image.
        """
        with torch.no_grad():
            if self.use_multiprocessing:
                # Use multiprocessing to preprocess images
                with Pool(cpu_count()) as pool:
                    inputs = pool.starmap(
                        self._process_image,
                        [(image, self.input_format, self.aug, self.cfg.MODEL.DEVICE) for image in images]
                    )
            else:
                # Sequential preprocessing
                inputs = [
                    self._process_image(image, self.input_format, self.aug, self.cfg.MODEL.DEVICE)
                    for image in images
                ]

            predictions = self.model(inputs)
            return predictions
