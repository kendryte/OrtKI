from abc import ABCMeta, abstractmethod

from attr import has
from onnx.defs import OpSchema
import io
import os

class LanguageTarget:
    __metaclass__ = ABCMeta
    def __init__(self, schema: OpSchema) -> None:
        self.schema = schema
        self.init_type_map()

    def op_name(self) -> str:
        return self.schema.name

    @abstractmethod
    def init_type_map(self) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    def output_type(self) -> str:
        if self.has_multi_output():
            return self.tensor_array()
        else:
            return self.tensor_type()

    def is_sequence(self, input_name) -> bool:
        return input_name.endswith("_sequence")

    @abstractmethod
    def output_path(self) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    @abstractmethod
    def tensor_type(self) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    @abstractmethod
    def size_type(self) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    @abstractmethod
    def array_type(self, ty) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    @abstractmethod
    def tensor_array(self) -> str:
        return self.array_type(self.tensor_type())

    @abstractmethod
    def codegen(self) -> str:
        raise NotImplementedError("Must override LanguageTarget")

    @abstractmethod
    def method_name(self) -> str:
        return self.interface_method_name()

    @abstractmethod
    def array_need_size(self) -> bool:
        raise NotImplementedError("Must override LanguageTarget")

    def interface_method_name(self) -> str:
        return f"ortki_{self.op_name()}"

    def map_attr_type(self, attrType: OpSchema.AttrType) -> str:
        return self.type_map[attrType]

    def process_input_param(self, input):
        # todo: maybe a bad design
        if self.is_array_input(input.name):
            array_param = f"{self.tensor_array()} {input.name}"
            if self.array_need_size():
                return f"{array_param}, {self.size_type()} input_size"
            else:
                return array_param
        else:
            return f"{self.tensor_type()} {input.name}"

    def process_attr_param(self, attr):
        attr_param = f"{self.map_attr_type(attr.type)} {attr.name}"
        if self.is_array_attr(attr) and self.array_need_size():
            return f"{attr_param}, {self.size_type()} {attr.name}_size"
        else:
            return attr_param

    def method_params(self):
        inputs = self.map_inputs(self.process_input_param)
        attrs = self.map_attrs(self.process_attr_param)
        return ', '.join(inputs + attrs)

    def method_sign(self, prefix = '') -> str:
        if prefix != '':
            prefix = prefix + ' '
        return f"{prefix}{self.output_type()} {self.method_name()}({self.method_params()})"

    def map_join_inputs(self, char, f):
        return char.join(self.map_inputs(f))

    def map_join_attrs(self, char, f):
        return char.join(self.map_attrs(f))

    def map_inputs(self, f):
        return list(map(f, self.schema.inputs))

    def map_attrs(self, f):
        return list(map(f, self.schema.attributes.values()))

    def has_multi_output(self) -> bool:
        # return len(self.schema.outputs) == 1 and self.schema.max_output > 1
        return self.schema.max_output > 1

    # when loop and ai.onnx.preview.training will be error, but loop has been blocked
    def is_array_input(self, input_name) -> bool:
        return (len(self.schema.inputs) == 1 and self.schema.max_input > 1) or self.is_sequence(input_name)

    def is_array_attr(self, attr) -> bool:
        return str(attr.type).endswith('S')

class CAPISRC(LanguageTarget):
    def __init__(self, schema: OpSchema) -> None:
        super().__init__(schema)

    def init_type_map(self):
        self.type_map = {
            OpSchema.AttrType.FLOAT: 'float',
            OpSchema.AttrType.INT: 'int64_t',
            OpSchema.AttrType.STRING: 'const char*',
            OpSchema.AttrType.TENSOR: self.tensor_type(),
            OpSchema.AttrType.GRAPH: 'int',
            # todo:this is error
            OpSchema.AttrType.SPARSE_TENSOR: self.tensor_type(),
            # AttributeError: type object 'onnx.onnx_cpp2py_export.defs.AttrType' has no attribute 'TYPE_PROTO'
            # OpSchema.AttrType.TYPE_PROTO: 'Error',
            OpSchema.AttrType.FLOATS: 'float*',
            OpSchema.AttrType.INTS: 'int64_t*',
            OpSchema.AttrType.STRINGS: 'const char**',
            OpSchema.AttrType.TENSORS: self.array_type(self.tensor_type()),
            OpSchema.AttrType.GRAPHS: 'Error',
            OpSchema.AttrType.SPARSE_TENSORS: 'Error',
            OpSchema.AttrType.TYPE_PROTO: 'int',
            OpSchema.AttrType.TYPE_PROTOS: 'int*',
        }

    def tensor_type(self) -> str:
        return "ortki::OrtKITensor *"

    def size_type(self) -> str:
        return "size_t"

    def gen_header(self) -> str:
        return f"{self.method_sign()};"

    def gen_source(self) -> str:
        return f"{self.method_sign()}\n{self.method_definition()}"

    def inputs_type(self) -> str:
        pass

    def output_type(self) -> str:
        if self.has_multi_output():
            return "ORTKI_API(ortki::OrtKITensorSeq *)"
        else:
            return f"ORTKI_API({self.tensor_type()})"

    def output_path(self) -> str:
        return os.path.join('..', 'include', 'operators.h')

    def array_need_size(self) -> bool:
        return True

    def array_type(self, ty) -> str:
        return f"{ty}*"

    def add_array_input(self, input_name: str) -> str:
        if self.is_sequence(input_name):
            return f"""
    {self.op_name()}.AddSeqInput("{input_name}", {input_name}, input_size);
"""
        else:
            return f"""for(int i = 0; i < input_size; ++i)
    {self.op_name()}.AddInput(std::string("{input_name}") + std::to_string(i), {input_name}[i]);
"""

    def add_single_input(self, input_name: str) -> str:
        return f"{self.op_name()}.AddInput(\"{input_name}\", {input_name});"

    def add_input(self, input):
        if self.is_array_input(input.name):
            return self.add_array_input(input.name)
        else:
            return self.add_single_input(input.name)

    def pass_attr(self, attr):
        if attr.type == OpSchema.AttrType.STRINGS:
            return f"ortki::ToVector<const char*, std::string>({attr.name}, {attr.name}_size)"
        elif self.is_array_attr(attr):
            return f"ortki::ToVector({attr.name}, {attr.name}_size)"
        elif attr.type == OpSchema.AttrType.TENSOR:
            return f"ortki::ToTensor({attr.name})"
        else:
            return attr.name

    def method_definition(self) -> str:
        def add_inputs(self) -> str:
            return self.map_join_inputs('\n', self.add_input)

        def add_attrs(self) -> str:
            def add_attr(attr):
                return f"{self.op_name()}.AddAttribute(\"{attr.name}\", {self.pass_attr(attr)});"
            return self.map_join_attrs('\n', add_attr)

        def compute_and_return(self) -> str:
            if self.has_multi_output():
                return f"return new ortki::OrtKITensorSeq({self.op_name()}.Run());"
                # return f"return ortki::fetches_to_tensors({self.op_name()}.Run());"
            else:
                return f"return new ortki::OrtKITensor({self.op_name()}.Run()[0]);"

        return "{"+ f"""
ortki::OpExecutor {self.op_name()}("{self.op_name()}");
{add_inputs(self)}
{add_attrs(self)}
{compute_and_return(self)}
""" + "}\n"


class CSharpGenerator:
    @staticmethod
    def root_path(f: str) -> str:
        return os.path.join('..', 'CSharp', 'OrtKISharp', f)

    @staticmethod
    def wrapper_class() -> str:
        return "OrtKI"

    @staticmethod
    def kernel_class() -> str:
        return "Native"

    @staticmethod
    def import_pkg():
        return ""

    @staticmethod
    def class_decl():
        return f"""internal partial class {CSharpGenerator.kernel_class()}\n"""

    @staticmethod
    def wrapper_class_decl():
        return f"""public partial class {CSharpGenerator.wrapper_class()}\n"""

    # todo:refactor
    @staticmethod
    def gen_class_source(using: str, body: str) -> str:
        return using + "namespace OrtKISharp;\n\n" + CSharpGenerator.class_decl() + "{\n" + body + "\n}"
        
    @staticmethod
    def gen_wrapper_class_source(using: str, body: str) -> str:
        return using + "namespace OrtKISharp;\n\n" + CSharpGenerator.wrapper_class_decl() + "{\n" + body + "\n}"


class CSharpDLLImport(LanguageTarget):
    def __init__(self, schema: OpSchema) -> None:
        super().__init__(schema)

    def init_type_map(self):
        self.type_map = {
            OpSchema.AttrType.FLOAT: 'float',
            OpSchema.AttrType.INT: 'long',
            OpSchema.AttrType.STRING: 'string',
            OpSchema.AttrType.TENSOR: self.tensor_type(),
            OpSchema.AttrType.GRAPH: 'int',
            # todo:this is error
            OpSchema.AttrType.SPARSE_TENSOR: self.tensor_type(),
            # AttributeError: type object 'onnx.onnx_cpp2py_export.defs.AttrType' has no attribute 'TYPE_PROTO'
            # OpSchema.AttrType.TYPE_PROTO: 'Error',
            OpSchema.AttrType.FLOATS: 'float[]',
            OpSchema.AttrType.INTS: 'long[]',
            OpSchema.AttrType.STRINGS: 'string[]',
            OpSchema.AttrType.TENSORS: 'IntPtr[]',
            OpSchema.AttrType.GRAPHS: 'int[]',
            OpSchema.AttrType.SPARSE_TENSORS: 'Error',
            OpSchema.AttrType.TYPE_PROTO: 'int',
            OpSchema.AttrType.TYPE_PROTOS: 'int[]',
        }

    def output_type(self) -> str:
        if self.has_multi_output():
            return self.tensorseq_type()
        else:
            return self.tensor_type()

    def array_need_size(self) -> bool:
        return True

    def tensor_type(self) -> str:
        return "Tensor"

    def tensorseq_type(self) -> str:
        return "TensorSeq"

    def size_type(self) -> str:
        return "nuint"

    def tensor_array(self) -> str:
        return "IntPtr[]"

    def make_decl(self) -> str:
        dll_import = '    [DllImport(LibraryName)]' + "\n" 
        decl = self.method_sign('    public static extern')
        return dll_import + decl

    def make_impl(self) -> str:
        return self.make_decl() + ';' + "\n"

    def array_type(self, ty) -> str:
        return f"{ty}[]"

class CSharpWrapper(CSharpDLLImport):
    def __init__(self, schema: OpSchema) -> None:
        super().__init__(schema)

    def tensor_type(self) -> str:
        return "Tensor"

    def size_type(self) -> str:
        return "nuint"

    def tensor_array(self) -> str:
        return self.array_type(self.tensor_type())

    def wrapper(self) -> str:
        return self.method_sign('    public static unsafe') + "\n{" + f"ortki_{self.op_name()}" + "}"

    def method_name(self) -> str:
        return self.op_name()

    def process_input_args(self, input):
        if self.is_array_input(input.name):
            return f"{input.name}.Select(x => x.DangerousGetHandle()).ToArray(), (nuint){input.name}.Length"
        else:
            return f"{input.name}"

    def process_attr_args(self, attr):
        if self.is_array_attr(attr):
            return f"{attr.name}, (nuint){attr.name}.Length"
        else:
            return f"{attr.name}"

    def array_need_size(self) -> bool:
        return False

    def output_type(self) -> str:
        return LanguageTarget.output_type(self)

    def select_array_attr(self):
        return filter(lambda attr: self.is_array_attr(attr), self.schema.attributes.values())

    def wrap_impl(self) -> str:
        def process_attr(attr):
            v = attr.name
            return v

        def method_args(self):
            inputs = self.map_inputs(self.process_input_args)
            attrs = self.map_attrs(self.process_attr_args)
            return ', '.join(inputs + attrs)

        def process_input_keep(input):
            if self.is_array_input(input.name):
                return f"        GC.KeepAlive({input.name});"
            else:
                return ""

        def keep_alives(self):
            keeps = self.map_inputs(process_input_keep)
            s = '\n'.join(list(filter(None, keeps)))
            return '\n' + s if s else ''

        def return_value(self) -> str:
            if self.has_multi_output():
                return "_tensor.ToTensorArray()"
            else:
                return "_tensor"

        return f"""        var _tensor = Native.{self.interface_method_name()}({method_args(self)});{keep_alives(self)}
        return {return_value(self)};"""

    def make_impl(self) -> str:
        return self.method_sign('    public static') + "\n    {\n" + self.wrap_impl() + "\n    }\n"

decl = "//This file is automatically generated from the onnx def files via tools/gen_operators.py.\n"

def capi(schemas) -> str:
    with io.open(os.path.join('..', 'include', 'operators.h'), 'w+', newline='', encoding="utf-8") as f:
        s = decl
        s += '#include "tensor.h"' + "\n"
        s += '#include "operators_patch.h"' + "\n"
        s += "\n".join([CAPISRC(schema).gen_header() for schema in schemas])
        f.write(s)

    with io.open(os.path.join('..', 'src', 'operators.cpp'), 'w+', newline='', encoding="utf-8") as f:
        s = decl
        s += '#include "operators.h"' + "\n" + '#include "op_executor.h"' + "\n\n"
        s += "\n".join([CAPISRC(schema).gen_source() for schema in schemas])
        f.write(s)

def cshapr_dll_import(schemas):
    with io.open(CSharpGenerator.root_path('Operators.cs'), 'w+', newline='', encoding="utf-8") as f:
        using = decl + "using System.Runtime.InteropServices;\n\n"
        s = "\n".join([CSharpDLLImport(schema).make_impl() for schema in schemas])
        f.write(CSharpGenerator.gen_class_source(using, s))

def csharp_wrapper(schemas):
    with io.open(CSharpGenerator.root_path('OperatorsWrapper.cs'), 'w+', newline='', encoding="utf-8") as f:
        using = decl
        s = "\n".join([CSharpWrapper(schema).make_impl() for schema in schemas])
        f.write(CSharpGenerator.gen_wrapper_class_source(using, s))

blockOps = ['If', 'Loop', 'Scan', 'Constant', 'ConstantOfShape', 'Split', 'Resize', 'BatchNormalization', 'Upsample', 'LSTM', 'Optional', 'SequenceMap']
targets = [
    capi, 
    cshapr_dll_import, 
    csharp_wrapper
]
