import os
import shutil
from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
jit_include_dirs = ('fp4_gemm/include/fp4_gemm', 'fp4_gemm/include/fp4_quant')
third_party_include_dirs = (
    'third-party/cutlass/include/cute',
    'third-party/cutlass/include/cutlass',
)
third_party_include_dir = 'third-party/cutlass/include'


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        self.make_jit_include_symlinks()

    @staticmethod
    def make_jit_include_symlinks():
        # Make symbolic links of third-party include directories
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = f'{current_dir}/{d}'
            dst_dir = f'{current_dir}/fp4_gemm/include/{dirname}'
            assert os.path.exists(src_dir)
            if os.path.exists(dst_dir):
                assert os.path.islink(dst_dir)
                os.unlink(dst_dir)
            os.symlink(src_dir, dst_dir, target_is_directory=True)


class CustomBuildPy(build_py):
    def run(self):
        # First, prepare the include directories
        self.prepare_includes()

        # Then run the regular build
        build_py.run(self)

    def prepare_includes(self):
        # Create temporary build directory instead of modifying package directory
        build_include_dir = os.path.join(self.build_lib, 'fp4_gemm/include')
        os.makedirs(build_include_dir, exist_ok=True)

        # Copy third-party includes to the build directory
        for d in third_party_include_dirs:
            dirname = d.split('/')[-1]
            src_dir = os.path.join(current_dir, d)
            dst_dir = os.path.join(build_include_dir, dirname)

            # Remove existing directory if it exists
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)

            # Copy the directory
            shutil.copytree(src_dir, dst_dir)


ext_modules = []

ext_modules.append(CUDAExtension(
    name="fp4_gemm",
    sources=[
        "fp4_gemm/include/fp4_quant/nvfp4_quant_cuda_kernels.cu",
        "fp4_gemm/include/fp4_quant/nvfp4_dequant_kernels.cu",
        "fp4_gemm/include/fp8_gemm/fp8_gemm.cu",
        "fp4_gemm/include/cuda_utils_kernels.cu",
        "fp4_gemm/include/torch_bindings.cpp"
    ],
    include_dirs=[],
    libraries=["m"],  # 链接数学库（如 -lm）
    extra_compile_args={
        "cxx": [
            "-std=c++17",
            "-O3",
            "-I./" + third_party_include_dir
        ],         # C++ 编译选项
        "nvcc": [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-I./" + third_party_include_dir
        ],  # NVCC 编译选项
    },
))


def get_requirements() -> list[str]:
    return []

if __name__ == '__main__':
    setup(
        name='fp4_gemm',
        version='1.0.0',
        install_requires=get_requirements(),
        packages=['fp4_gemm', 'fp4_gemm/jit', 'fp4_gemm/jit_kernels'],
        package_data={
            'fp4_gemm': [
                'include/fp4_gemm/*',
                'include/cute/**/*',
                'include/cutlass/**/*',
            ]
        },
        ext_modules=ext_modules,
        cmdclass={
            # "develop": PostDevelopCommand,
            "build_ext": BuildExtension,
        },  # 必须添加以支持混合编译
    )
