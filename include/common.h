#pragma once

namespace ort_ki
{
#define DEFAULT_OPSET 15

#ifdef _WIN32
    #include <intrin.h>
#define ORTKI_API(ret) extern "C" __declspec(dllexport) ret
#else
#define ORTKI_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif

}
