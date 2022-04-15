from detectron2.utils.visualizer import _create_text_labels, Visualizer, GenericMask, ColorMode
import numpy as np


class MySegVisualizer(Visualizer):
    def draw_instance_segmentations_on_predictions(self, predictions):
        """
        Identical to Visualizer.draw_instance_predictions except only draws segmentations in fixed white color
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=None,
            labels=None,
            keypoints=keypoints,
            assigned_colors=[np.array([[0, 1, 0]])],
            alpha=0.3,
        )
        return self.output

