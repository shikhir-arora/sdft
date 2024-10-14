import sys
import os
import subprocess
import shutil
import tempfile
import shlex
from setuptools import Extension
import numpy

def get_default_compiler():
    compilers = ['clang', 'gcc', 'cc']
    for compiler in compilers:
        if subprocess.call(shlex.split(f'{compiler} --version'), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
            return compiler
    return None

class OpenMPSetup:
    def install_libomp(self):
        try:
            brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
            subprocess.check_call(['brew', 'install', 'libomp'])
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Homebrew not found. Please install Homebrew or libomp manually.")
            sys.exit(1)

    def compiler_has_openmp(self):
        default_compiler = get_default_compiler()
        if not default_compiler:
            print("No C compiler found. Please install one.")
            return False

        tmp_dir = tempfile.mkdtemp()
        file_path = os.path.join(tmp_dir, 'test_openmp.c')
        exec_path = os.path.join(tmp_dir, 'test_openmp')

        program = '''
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
        '''

        with open(file_path, 'w') as f:
            f.write(program)

        compile_command = f'{default_compiler} -o {exec_path} {file_path} -fopenmp'
        
        try:
            subprocess.run(compile_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            result = subprocess.run(exec_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except subprocess.CalledProcessError:
            return False
        finally:
            shutil.rmtree(tmp_dir)

    def get_openmp_flags(self):
        compile_flags = []
        link_flags = []
        has_openmp = self.compiler_has_openmp()

        if has_openmp:
            if sys.platform == 'win32':
                compile_flags += ['/openmp', '/O2']
            else:
                compile_flags += ['-O3', '-march=native', '-funroll-loops', '-fopenmp']
                link_flags += ['-fopenmp']
                if sys.platform == 'darwin':
                    try:
                        brew_prefix = subprocess.check_output(['brew', '--prefix'], text=True).strip()
                        omp_include = os.path.join(brew_prefix, 'include')
                        omp_lib = os.path.join(brew_prefix, 'lib')
                        compile_flags += [f'-I{omp_include}']
                        link_flags += [f'-L{omp_lib}', '-lomp']
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        print("Warning: Homebrew not found. OpenMP support may be limited.")
        else:
            if sys.platform == 'darwin':
                print("libomp not found. Attempting to install via Homebrew...")
                self.install_libomp()
                return self.get_openmp_flags()
            else:
                print("Warning: OpenMP not available. Parallel code will run sequentially.")
                if sys.platform == 'win32':
                    compile_flags += ['/O2']
                else:
                    compile_flags += ['-O3', '-march=native', '-funroll-loops']

        return {'compile_flags': compile_flags, 'link_flags': link_flags}

def build(setup_kwargs):
    omp_setup = OpenMPSetup()
    openmp_flags = omp_setup.get_openmp_flags()

    numpy_includes = numpy.get_include()

    extensions = [
        Extension(
            name="cython_sdft_functions",
            sources=["cython_sdft_functions.pyx"],
            include_dirs=[numpy_includes],
            define_macros=[("CYTHON_WITHOUT_ASSERTIONS", "1")],
            extra_compile_args=openmp_flags['compile_flags'],
            extra_link_args=openmp_flags['link_flags'],
            language='c'
        )
    ]

    setup_kwargs.update({
        'ext_modules': extensions,
    })

# debug
if __name__ == "__main__":
    setup_kwargs = {}
    build(setup_kwargs)
    print(setup_kwargs)