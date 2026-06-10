"""
Semua arsitektur model: CNN variants, MobileNetV2, HOG+SVM.
Setiap builder mengembalikan model siap-compile.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def build_cnn(filters, dense_units, num_classes, input_shape=(32, 32, 1),
              use_augmentation=False, aug_config=None, dropout_conv=0.2,
              dropout_dense=0.4):
    """
    Build custom CNN dengan jumlah filter dan dense units yang dikonfigurasi.

    Parameters
    ----------
    filters : list[int]
        Jumlah filter per blok konvolusi, e.g. [16, 32, 64].
    dense_units : int
        Jumlah unit pada dense classifier head.
    num_classes : int
    input_shape : tuple
    use_augmentation : bool
        Apakah menambahkan lapisan augmentasi spasial mikro.
    aug_config : dict or None
        Konfigurasi augmentasi {rotation_range, translation_range}.
    """
    from tensorflow.keras import layers, models

    layer_list = [layers.Input(shape=input_shape)]

    # Augmentasi spasial mikro (hanya aktif saat training)
    if use_augmentation and aug_config:
        layer_list.append(
            layers.RandomRotation(
                aug_config.get("rotation_range", 0.02),
                fill_mode="constant", fill_value=0.0,
            )
        )
        layer_list.append(
            layers.RandomTranslation(
                height_factor=aug_config.get("translation_range", 0.04),
                width_factor=aug_config.get("translation_range", 0.04),
                fill_mode="constant", fill_value=0.0,
            )
        )

    # Blok konvolusi
    for i, n_filters in enumerate(filters):
        layer_list.append(
            layers.Conv2D(n_filters, (3, 3), activation="relu", padding="same")
        )
        layer_list.append(layers.BatchNormalization())
        layer_list.append(layers.MaxPooling2D((2, 2)))
        drop_rate = 0.3 if i == len(filters) - 1 else dropout_conv
        layer_list.append(layers.Dropout(drop_rate))

    # Classifier head
    layer_list.append(layers.Flatten())
    layer_list.append(layers.Dense(dense_units, activation="relu"))
    layer_list.append(layers.BatchNormalization())
    layer_list.append(layers.Dropout(dropout_dense))
    layer_list.append(layers.Dense(num_classes, activation="softmax"))

    model = models.Sequential(layer_list)
    return model


def build_standard_cnn(num_classes, input_shape=(32, 32, 1)):
    """Model standar industri: 32-64-128 filters, Dense 256."""
    return build_cnn(
        filters=[32, 64, 128],
        dense_units=256,
        num_classes=num_classes,
        input_shape=input_shape,
        use_augmentation=False,
    )


def build_hybrid_cnn(num_classes, input_shape=(32, 32, 1), aug_config=None):
    """Model hybrid skeleton: 16-32-64 filters, Dense 128 + augmentasi mikro."""
    return build_cnn(
        filters=[16, 32, 64],
        dense_units=128,
        num_classes=num_classes,
        input_shape=input_shape,
        use_augmentation=True,
        aug_config=aug_config or {"rotation_range": 0.02, "translation_range": 0.04},
    )


def build_iso_parameter_cnn(num_classes, input_shape=(32, 32, 1)):
    """
    Iso-parameter CNN: arsitektur identik dengan hybrid (16-32-64, Dense 128)
    tetapi tanpa augmentasi (untuk input raw).
    ~163k parameter — mengisolasi efek skeletonization dari kapasitas model.
    """
    return build_cnn(
        filters=[16, 32, 64],
        dense_units=128,
        num_classes=num_classes,
        input_shape=input_shape,
        use_augmentation=False,
    )


def build_mobilenetv2(num_classes, input_shape=(32, 32, 1), alpha=0.35):
    """
    MobileNetV2 lightweight: alpha=0.35 untuk perbandingan SOTA ringan.
    Input grayscale di-tile ke 3 channel.
    """
    from tensorflow.keras import layers, models
    import tensorflow as tf

    inputs = layers.Input(shape=input_shape)

    # Grayscale → 3 channel (MobileNetV2 expects 3ch)
    x = layers.Concatenate()([inputs, inputs, inputs])

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(input_shape[0], input_shape[1], 3),
        alpha=alpha,
        include_top=False,
        weights=None,  # Train from scratch
        pooling="avg",
    )

    x = base_model(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)


def build_hog_svm(config):
    """
    HOG + SVM pipeline (metode klasik).
    Returns sklearn Pipeline.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from skimage.feature import hog

    svm_cfg = config["experiments"]["baselines"]["models"]["hog_svm"]

    class HOGTransformer:
        """Custom sklearn transformer untuk ekstraksi fitur HOG."""
        def __init__(self, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2)):
            self.orientations = orientations
            self.pixels_per_cell = tuple(pixels_per_cell)
            self.cells_per_block = tuple(cells_per_block)

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            features = []
            for img in X:
                img_2d = img.squeeze()
                feat = hog(
                    img_2d,
                    orientations=self.orientations,
                    pixels_per_cell=self.pixels_per_cell,
                    cells_per_block=self.cells_per_block,
                    feature_vector=True,
                )
                features.append(feat)
            return np.array(features)

    pipeline = Pipeline([
        ("hog", HOGTransformer(
            orientations=svm_cfg.get("hog_orientations", 9),
            pixels_per_cell=svm_cfg.get("hog_pixels_per_cell", [8, 8]),
            cells_per_block=svm_cfg.get("hog_cells_per_block", [2, 2]),
        )),
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=svm_cfg.get("svm_kernel", "rbf"),
            C=svm_cfg.get("svm_C", 10.0),
            probability=True,
            random_state=config["project"]["random_seed"],
        )),
    ])

    return pipeline


def compile_keras_model(model, config):
    """Compile Keras model dengan konfigurasi dari config.yaml."""
    model.compile(
        optimizer=config["training"]["optimizer"],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def count_params(model):
    """Hitung total parameter (trainable + non-trainable)."""
    if hasattr(model, "count_params"):
        return model.count_params()
    # sklearn model
    return 0
