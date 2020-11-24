#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

// for the unpooling modules, we have in_mpoint<out_npoint
REGISTER_OP("MeshInterpolate")
    .Input("input: float32")      // batch_mpoints * in_channels
    .Input("vt_replace: int32")   // batch_npoints
    .Input("vt_map: int32")       // batch_npoints
    .Output("output: float32")    // batch_npoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle input, vt_replace;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &vt_replace));
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(vt_replace, 0),
                                                                          c->Dim(input, 1)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("MeshInterpolateGrad")
    .Input("input: float32")        // batch_mpoints * in_channels
    .Input("grad_output: float32")  // batch_npoints * in_channels
    .Input("vt_replace: int32")     // batch_npoints
    .Input("vt_map: int32")         // batch_npoints
    .Output("grad_input: float32")  // batch_mpoints * in_channels
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });


void interpolateLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                             const float* input, float* output);
class InterpolateGpuOp : public OpKernel {
    public:
        explicit InterpolateGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& input_tensor     = context->input(0);
            const Tensor& vtReplace_tensor = context->input(1);
            const Tensor& vtMap_tensor     = context->input(2);

            // get the dims required by computations, (nvIn<nvOut)
            int nvIn  = input_tensor.shape().dim_size(0);   // number of input points
            int C     = input_tensor.shape().dim_size(1);   // number of input channels
            int nvOut = vtMap_tensor.shape().dim_size(0);   // number of output points

            OP_REQUIRES(context, input_tensor.dims()==2, errors::InvalidArgument(
            "rank of input should be 2, i.e. (batch_mpoints,channels)"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtReplace_tensor.shape()),
                errors::InvalidArgument("vtReplace expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat     = input_tensor.flat<float>();
            auto vtReplace_flat = vtReplace_tensor.flat<int32>();
            auto vtMap_flat     = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* output_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvOut,C}, &output_tensor));
            auto output_flat = output_tensor->flat<float>();

            const float* input   = &(input_flat(0));
            const int* vtReplace = &(vtReplace_flat(0));
            const int* vtMap     = &(vtMap_flat(0));

            float* output = &(output_flat(0));
            cudaMemset(output, 0, sizeof(float)*nvOut*C);
            interpolateLauncher(nvIn, C, nvOut, vtReplace, vtMap, input, output);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshInterpolate").Device(DEVICE_GPU), InterpolateGpuOp);


void interpolateGradLauncher(int nvIn, int C, int nvOut, const int* vtReplace, const int* vtMap,
                                 const float* gradOutput, float* gradInput);
class InterpolateGradGpuOp : public OpKernel {
    public:
        explicit InterpolateGradGpuOp(OpKernelConstruction* context) : OpKernel(context){}

        void Compute(OpKernelContext* context) override
        {
            // Grab the input tensors
            const Tensor& input_tensor       = context->input(0);
            const Tensor& grad_output_tensor = context->input(1);
            const Tensor& vtReplace_tensor   = context->input(2);
            const Tensor& vtMap_tensor       = context->input(3);

            // get the dims required by computations, (nvIn<nvOut)
            int nvIn  = input_tensor.shape().dim_size(0);   // number of input points
            int C     = input_tensor.shape().dim_size(1);   // number of input channels
            int nvOut = vtMap_tensor.shape().dim_size(0);   // number of output points

            OP_REQUIRES(context, input_tensor.dims()==2, errors::InvalidArgument(
            "rank of input should be 2, i.e. (batch_mpoints,channels)"));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtReplace_tensor.shape()),
                errors::InvalidArgument("vtReplace expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("vtMap expects a 1-D vector."));

            // flatten the input tensors
            auto input_flat       = input_tensor.flat<float>();
            auto grad_output_flat = grad_output_tensor.flat<float>();
            auto vtReplace_flat   = vtReplace_tensor.flat<int32>();
            auto vtMap_flat       = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* grad_input_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nvIn,C}, &grad_input_tensor));
            auto grad_input_flat = grad_input_tensor->flat<float>();

            const float* input   = &(input_flat(0));
            const float* gradOut = &(grad_output_flat(0));
            const int* vtReplace = &(vtReplace_flat(0));
            const int* vtMap     = &(vtMap_flat(0));

            float* gradIn = &(grad_input_flat(0));
            cudaMemset(gradIn, 0, sizeof(float)*nvIn*C);
            interpolateGradLauncher(nvIn, C, nvOut, vtReplace, vtMap, gradOut, gradIn);
        }
};
REGISTER_KERNEL_BUILDER(Name("MeshInterpolateGrad").Device(DEVICE_GPU), InterpolateGradGpuOp);


