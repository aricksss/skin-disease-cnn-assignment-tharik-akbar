"""
Model architecture definitions for Skin Disease CNN Classification.
Implements:
1. MobileNetV2
2. ResNet50
3. DenseNet121

Each model utilizes ImageNet pretrained weights, frozen backbone,
GlobalAveragePooling2D, regularization (Dropout/BatchNorm), and Softmax output.
"""

import sys
from pathlib import Path
import tensorflow as tf

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGE_SIZE, NUM_CLASSES, INITIAL_LR

def build_mobilenetv2(num_classes: int = NUM_CLASSES, learning_rate: float = INITIAL_LR) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model with ImageNet weights."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batchnorm")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_resnet50(num_classes: int = NUM_CLASSES, learning_rate: float = INITIAL_LR) -> tf.keras.Model:
    """Build ResNet50 transfer learning model with ImageNet weights."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batchnorm")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ResNet50_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_densenet121(num_classes: int = NUM_CLASSES, learning_rate: float = INITIAL_LR) -> tf.keras.Model:
    """Build DenseNet121 transfer learning model with ImageNet weights."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    
    base_model = tf.keras.applications.DenseNet121(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_batchnorm")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense_1")(x)
    x = tf.keras.layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="DenseNet121_Transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def get_model_summary_info(model: tf.keras.Model) -> dict:
    """Extract parameter count statistics."""
    total_params = model.count_params()
    trainable_params = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    non_trainable_params = sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    return {
        "model_name": model.name,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params)
    }

if __name__ == "__main__":
    print("Testing Model Instantiations...")
    for builder in [build_mobilenetv2, build_resnet50, build_densenet121]:
        m = builder()
        info = get_model_summary_info(m)
        print(f"[{info['model_name']}] Total: {info['total_parameters']:,} | Trainable: {info['trainable_parameters']:,} | Non-trainable: {info['non_trainable_parameters']:,}")
