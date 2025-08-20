use crate::storage::HasStorage;
use crate::storage::metal_gpu::{MetalGpu, MetalGpuStorage};
use crate::tensor_ops::relu::Relu;
use objc2::AnyThread;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixNeuron, MPSCNNNeuronType,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use std::mem;

impl Relu<f32> for MetalGpu {
    fn relu<const N: usize>(
        input: &<Self as HasStorage<f32, N>>::Storage,
        output: &mut <Self as HasStorage<f32, N>>::Storage,
    ) where
        Self: HasStorage<f32, N>,
    {
        unsafe {
            // Get raw buffer handles
            let storage_input: &MetalGpuStorage = &*(input as *const _ as *const _);
            let storage_output: &MetalGpuStorage = &*(output as *const _ as *mut _);

            let device = &MetalGpu::shared().device;
            let queue = &MetalGpu::shared().queue;

            // Create matrix descriptors for input and output
            let element_size = mem::size_of::<f32>();
            let row_bytes = N * element_size;
            
            let desc_input = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                N,        // rows
                1,        // columns (treat as Nx1 matrix)
                row_bytes,
                MPSDataType::Float32,
            );
            
            let desc_output = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                N,        // rows
                1,        // columns
                row_bytes,
                MPSDataType::Float32,
            );

            // Create MPSMatrix objects from the buffers
            let mat_input = MPSMatrix::initWithBuffer_descriptor(
                MPSMatrix::alloc(),
                &storage_input.buffer,
                &desc_input,
            );
            
            let mat_output = MPSMatrix::initWithBuffer_descriptor(
                MPSMatrix::alloc(),
                &storage_output.buffer,
                &desc_output,
            );

            // Create the ReLU neuron - use the correct constructor
            let relu = MPSMatrixNeuron::initWithDevice(
                MPSMatrixNeuron::alloc(),
                device,
            );
            
            // Set the neuron type to ReLU
            relu.setNeuronType_parameterA_parameterB_parameterC(
                MPSCNNNeuronType::ReLU,
                0.0, // parameterA for ReLU
                0.0, // parameterB
                0.0, // parameterC
            );

            // Execute the ReLU operation
            let cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
                queue.commandBuffer().expect("expected a command buffer");
            
            relu.encodeToCommandBuffer_inputMatrix_biasVector_resultMatrix(
                &cmd_buf,
                &mat_input,
                None,  // No bias vector for ReLU
                &mat_output,
            );
            
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
    }

    fn relu_backward<const N: usize>(
        _input: &<Self as HasStorage<f32, N>>::Storage,
        _grad_output: &<Self as HasStorage<f32, N>>::Storage,
        _grad_input: &mut <Self as HasStorage<f32, N>>::Storage,
    ) where
        Self: HasStorage<f32, N>,
    {
        // TODO: Implement GPU relu_backward using MPSMatrixNeuronGradient
        unimplemented!("GPU relu_backward not implemented yet");
    }
}
