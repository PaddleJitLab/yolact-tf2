import tensorflow as tf
from nets.yolact import get_train_model, yolact

model_body = yolact([544, 544, 3], 100, train_mode = True)
model = get_train_model(model_body)

# x = tf.random.Generator.from_seed(1).normal(shape=[544, 544, 3])
# print(model(x))

try:
    tf.saved_model.save(model, './')
    print("[JIT] Export model by TensorFlow successed.")
except:
    print("[JIT] Export model by TensorFlow failed.")