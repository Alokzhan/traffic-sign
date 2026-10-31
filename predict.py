"""Inference + Grad-CAM.  python predict.py --model outputs/final.keras --image stop.png"""
import argparse, cv2, numpy as np, tensorflow as tf
from tensorflow import keras
from common import *


def gradcam(model, x):
    conv = model.get_layer("last_conv")
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
    ap.add_argument("--model", required=True); ap.add_argument("--image", required=True)
    ap.add_argument("--threshold", type=float, default=0.70)
    ap.add_argument("--cam", default="gradcam.png")
    a = ap.parse_args()
    model = keras.models.load_model(a.model)
    img = cv2.imread(a.image)
    assert img is not None, "Image read nahi hui"
    x = preprocess(img)[None].astype("float32")
    p = model.predict(x, verbose=0)[0]
    top = p.argsort()[::-1][:3]
    for i in top:
        print(f"{CLASS_NAMES[i]:20s} {p[i]:.1%}")
    print("-> UNCERTAIN (confidence kam)" if p[top[0]] < a.threshold else f"-> {CLASS_NAMES[top[0]]}")
    cam = cv2.resize(gradcam(model, x), (192, 192))
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    base = cv2.resize(cv2.cvtColor(x[0].astype("uint8"), cv2.COLOR_RGB2BGR), (192, 192))
    cv2.imwrite(a.cam, cv2.addWeighted(base, 0.6, heat, 0.4, 0))
    print("Grad-CAM saved:", a.cam, "\n" + DISCLAIMER)


if __name__ == "__main__":
    main()
