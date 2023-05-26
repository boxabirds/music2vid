import tensorflow as tf

print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available (deprecated):", tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
print("GPU Available (new):", tf.config.list_physical_devices('GPU'))

# Check if TensorFlow is built with CuDNN support
if tf.test.is_built_with_cuda():
    from tensorflow.python.platform import build_info as tf_build_info
    print("Built with CuDNN:", tf_build_info.build_info['cudnn_version'] is not None)
    print("CuDNN version:", tf_build_info.build_info['cudnn_version'])
    print("CUDA version:", tf_build_info.build_info['cuda_version'])
