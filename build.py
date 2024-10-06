import sys
import os
import subprocess
import shutil
import tempfile
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        openmp_flags = self.get_openmp_flags(compiler_type)

        for ext in self.extensions:
            ext.extra_compile_args += openmp_flags['compile_flags']
            ext.extra_link_args += openmp_flags['link_flags']

        build_ext.build_extensions(self)

    def get_openmp_flags(self, compiler_type):
        # Default flags
        compile_flags = []
        link_flags = []

        has_openmp = self.compiler_has_openmp()

        if has_openmp:
            if compiler_type == 'msvc':
                # Microsoft Visual C++
                compile_flags += ['/openmp', '/O2']
            elif compiler_type == 'unix':
                # Unix-like systems (Linux, macOS)
                compile_flags += ['-O3', '-march=native', '-funroll-loops', '-fopenmp']
                link_flags += ['-fopenmp']
                if sys.platform == 'darwin':
                    # macOS specific adjustments
                    brew_prefix = self.get_brew_prefix()
                    if brew_prefix:
                        omp_include = os.path.join(brew_prefix, 'include')
                        omp_lib = os.path.join(brew_prefix, 'lib')
                        compile_flags += [f'-I{omp_include}']
                        link_flags += [f'-L{omp_lib}', '-lomp']
                    else:
                        print("Error: Homebrew not found. Please install Homebrew or libomp manually.")
                        sys.exit(1)
            else:
                # Other compilers
                compile_flags += ['-fopenmp']
                link_flags += ['-fopenmp']
        else:
            if sys.platform == 'darwin':
                # Attempt to install libomp via Homebrew
                print("libomp not found. Attempting to install via Homebrew...")
                self.install_libomp()
                # Retry OpenMP detection
                has_openmp = self.compiler_has_openmp()
                if not has_openmp:
                    print("Error: Could not install libomp. Please install libomp manually.")
                    sys.exit(1)
                else:
                    return self.get_openmp_flags(compiler_type)
            else:
                print("Warning: OpenMP not available. Parallel code will run sequentially.")
                if compiler_type == 'msvc':
                    compile_flags += ['/O2']
                else:
                    compile_flags += ['-O3', '-march=native', '-funroll-loops']

        return {'compile_flags': compile_flags, 'link_flags': link_flags}

    def compiler_has_openmp(self):
        tmp_dir = tempfile.mkdtemp()
        filename = 'test_openmp.c'
        file_path = os.path.join(tmp_dir, filename)

        program = """
        #include <omp.h>
        int main(void) {
            int nthreads = 0;
            #pragma omp parallel
            {
                #pragma omp atomic
                nthreads += 1;
            }
            return nthreads > 0 ? 0 : 1;
        }
        """
        with open(file_path, 'w') as f:
            f.write(program)

        try:
            # Compile the test program
            compiler = self.compiler
            objects = compiler.compile([file_path], output_dir=tmp_dir)
            # Link the test program
            compiler.link_executable(objects, 'test_openmp', output_dir=tmp_dir)
            # Run the test program
            exec_path = os.path.join(tmp_dir, 'test_openmp' + ('.exe' if sys.platform == 'win32' else ''))
            result = subprocess.run([exec_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            has_openmp = result.returncode == 0
        except Exception:
            has_openmp = False
        finally:
            shutil.rmtree(tmp_dir)

        return has_openmp

    def install_libomp(self):
        brew_prefix = self.get_brew_prefix()
        if brew_prefix:
            try:
                subprocess.check_call(['brew', 'install', 'libomp'])
            except subprocess.CalledProcessError:
                print("Error: Failed to install libomp via Homebrew.")
                sys.exit(1)
        else:
            print("Error: Homebrew not found. Please install Homebrew or libomp manually.")
            sys.exit(1)

    def get_brew_prefix(self):
        try:
            brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
            return brew_prefix
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

# Get numpy include directory
numpy_includes = numpy.get_include()

extensions = [
    Extension(
        name="cython_sdft_functions",
        sources=["cython_sdft_functions.pyx"],
        include_dirs=[numpy_includes],
        define_macros=[("CYTHON_WITHOUT_ASSERTIONS", "1")],
        language='c'
    )
]

setup(
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExt},
)
