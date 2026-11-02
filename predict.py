"""Inference + Grad-CAM.  python predict.py --model outputs/final.keras --image stop.png"""
import argparse, cv2, numpy as np, tensorflow as tf
from tensorflow import keras
from common import CLASS_NAMES, preprocess, DISCLAIMER


def gradcam(model, x):
    try:
        conv = model.get_layer("last_conv")
    except Exception:
        conv_layers = [l for l in model.layers if "conv" in l.name.lower()]
        if not conv_layers:
            return np.ones((x.shape[1], x.shape[2]), dtype="float32")
        conv = conv_layers[-1]
    gm = keras.Model(model.inputs, [conv.output, model.output])
    with tf.GradientTape() as t:
        fmap, preds = gm(x)
        loss = preds[:, tf.argmax(preds[0])]
    grads = t.gradient(loss, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(fmap * w[:, None, None, :], axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--cam", default="gradcam.png")
    a = ap.parse_args()

    model = keras.models.load_model(a.model)
    img = cv2.imread(a.image)
    assert img is not None, f"Image read nahi hui: {a.image}"

    x = preprocess(img)[None].astype("float32")
    p = model.predict(x, verbose=0)[0]
    n_out = len(p)
    classes = CLASS_NAMES[:n_out]

    top = p.argsort()[::-1][:3]
    print(f"\nTop predictions for {a.image}:")
    for i in top:
        print(f"  {classes[i]:25s} {p[i]:.1%}")

    best = top[0]
    if "Other" in classes[best]:
        print(f"\n-> {classes[best]} (Not one of the supported traffic signs)")
    elif p[best] < a.threshold:
        print(f"\n-> UNCERTAIN (confidence {p[best]:.1%} is below threshold {a.threshold:.0%})")
    else:
        print(f"\n-> {classes[best]} (confidence {p[best]:.1%})")

    cam = cv2.resize(gradcam(model, x), (192, 192))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    base = cv2.resize(cv2.cvtColor(x[0].astype("uint8"), cv2.COLOR_RGB2BGR), (192, 192))
    cv2.imwrite(a.cam, cv2.addWeighted(base, 0.6, heat, 0.4, 0))
    print("Grad-CAM saved:", a.cam)
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
