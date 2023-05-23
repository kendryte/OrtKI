#pragma once
#include <cstdint>
#include <cstddef>

namespace ortki
{
// https://onnxruntime.ai/docs/reference/compatibility.html
// ortki/onnxruntime/VERSION_NUMBER
#define DEFAULT_OPSET 16

#ifdef _WIN32
#include <intrin.h>
#define ORTKI_API(ret) extern "C" __declspec(dllexport) ret
#else
#define ORTKI_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif
}
