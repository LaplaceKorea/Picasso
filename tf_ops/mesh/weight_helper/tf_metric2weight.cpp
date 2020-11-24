#include <cmath> // sqrtf
#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;


REGISTER_OP("NormalizeWeight")
  .Input("weight_in: float32")    // concat_nvIn: (raw weight of each vertex to its cluster center)
  .Input("vt_rep: int32")         // concat_nvIn: (vertex replacement: clustering information)
  .Input("vt_map: int32")         // concat_nvIn: (vertex mapping: map input to output vertices)
  .Output("weight_out: float32")  // concat_nvIn: (normalized weight of each vertex to its cluster center)
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // output shape setting
    c->set_output(0, c->input(0));
    return Status::OK();
  });


void normalizeWeightLauncher(const int NvIn, const float* wgtIn, const int* vtReplace,
                             const int* vtMap, float* wgtOut);
class NormalizeWeightGpuOp: public OpKernel{
    public:
        explicit NormalizeWeightGpuOp(OpKernelConstruction* context):OpKernel(context) {}

        void Compute(OpKernelContext * context) override {
            // Grab the input tensors
            const Tensor& wgtIn_tensor = context->input(0);
            const Tensor& vtRep_tensor = context->input(1);
            const Tensor& vtMap_tensor = context->input(2);

            // get the dims required by computations
            int NvIn = wgtIn_tensor.shape().dim_size(0);    // number of input vertices/points

            // conditional checks and validation
            OP_REQUIRES(context, wgtIn_tensor.shape().dim_size(0)==vtRep_tensor.shape().dim_size(0),
                        errors::InvalidArgument("The shape of vtReplace should equal input weight."));
            OP_REQUIRES(context, wgtIn_tensor.shape().dim_size(0)==vtMap_tensor.shape().dim_size(0),
                       errors::InvalidArgument("The shape of vtMap should equal input weight."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(wgtIn_tensor.shape()),
                errors::InvalidArgument("rep expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtRep_tensor.shape()),
                errors::InvalidArgument("rep expects a 1-D vector."));
            OP_REQUIRES(context, TensorShapeUtils::IsVector(vtMap_tensor.shape()),
                errors::InvalidArgument("map expects a 1-D vector."));

            // flatten the input tensors
            auto wgtIn_flat = wgtIn_tensor.flat<float>();
            auto vtRep_flat = vtRep_tensor.flat<int32>();
            auto vtMap_flat = vtMap_tensor.flat<int32>();

            // Create an output tensor
            Tensor* wgtOut_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{NvIn}, &wgtOut_tensor));
            auto wgtOut_flat = wgtOut_tensor->flat<float>();

            const float* wgtIn = &(wgtIn_flat(0));
            const int*   vtRep = &(vtRep_flat(0));
            const int*   vtMap = &(vtMap_flat(0));

            float* wgtOut = &(wgtOut_flat(0));
            cudaMemset(wgtOut,0,sizeof(float)*NvIn); // initialize dist all to zeros
            normalizeWeightLauncher(NvIn, wgtIn, vtRep, vtMap, wgtOut);
        }
};
REGISTER_KERNEL_BUILDER(Name("NormalizeWeight").Device(DEVICE_GPU),NormalizeWeightGpuOp);





