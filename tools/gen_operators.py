#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import io
import os
import sys

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple
import generator
from onnx.onnx_cpp2py_export.defs import get_all_schemas

SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = False


ext = '.cpp'


def display_schema(schema, versions, target):  # type: (OpSchema, Sequence[OpSchema], type) -> Text

    return generator.CAPISRC(schema).gen_source()

def operator_schemas():
    output = []
    # domain -> support level -> name -> [schema]
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
    existing_ops = set()  # type: Set[Text]
    for domain, _supportmap in sorted(index.items()):
        # if not should_render_domain(domain):
        #     continue

        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in existing_ops:
                    continue
                existing_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    # todo:which should be return
    for domain, supportmap in operator_schemas:
        for _, namemap in supportmap:
            for op_type, schema, versions in namemap:
                if not schema.name in generator.blockOps:
                    output.append(schema)
    return output

if __name__ == '__main__':
    scheams = operator_schemas()
    for gen in generator.targets:
        gen(scheams)
