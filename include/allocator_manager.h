#pragma once
#include "core/framework/allocator.h"

using namespace onnxruntime;
namespace ort_ki {
    class AllocatorManager {
    public:
        // the allocator manager is a just for onnx runner to allocate space for input/output tensors.
        // onnxruntime session will use the allocator owned by execution provider.
        static AllocatorManager &Instance();

        /**
        Destruct th AllocatorManager. Will unset Instance().
        */
        ~AllocatorManager();

        AllocatorPtr GetAllocator(const std::string &name, const int id = 0, bool arena = true);

    private:
        ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AllocatorManager);

        AllocatorManager();

        Status InitializeAllocators();

        std::unordered_map<std::string, AllocatorPtr> map_;
    };
}