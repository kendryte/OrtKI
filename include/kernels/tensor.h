#pragma once
#include "default_providers.h"
#include "../tensor.h"
#include "common.h"

namespace ortki
{
    ORTKI_API(OrtKITensor*) ortki_Cast(OrtKITensor *input, DataType dataType);

}