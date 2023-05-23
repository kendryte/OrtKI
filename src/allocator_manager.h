#pragma once
#include <onnxruntime/core/framework/allocator.h>

namespace ortki
{
    class AllocatorManager
    {
    public:
        // the allocator manager is a just for onnx runner to allocate space for input/output tensors.
        // onnxruntime session will use the allocator owned by execution provider.
        static AllocatorManager &Instance();

        /**
        Destruct th AllocatorManager. Will unset Instance().
        */
        ~AllocatorManager();

        onnxruntime::AllocatorPtr GetAllocator(const std::string &name, const int id = 0, bool arena = true);

    private:
        ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AllocatorManager);

        AllocatorManager();

        onnxruntime::Status InitializeAllocators();

        std::unordered_map<std::string, onnxruntime::AllocatorPtr> map_;
    };
}