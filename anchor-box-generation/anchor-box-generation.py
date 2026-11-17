def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    anchors = []

    stride = image_size / feature_size

    for i in range(feature_size):
        for j in range(feature_size):

            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride

            for scale in scales:
                for ratio in aspect_ratios:

                    w = scale * (ratio ** 0.5)
                    h = scale / (ratio ** 0.5)

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    anchors.append([x1, y1, x2, y2])

    return anchors