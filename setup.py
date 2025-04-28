from setuptools import setup, find_packages
from setuptools.command.build import build
from setuptools_scm.version import get_local_dirty_tag
import subprocess
from pathlib import Path
from os.path import join
import sys


def check_artifact_out_of_date(artifact, origin):
    art_path = Path(artifact)
    org_path = Path(origin)

    if not art_path.is_file():
        return True

    return art_path.stat().st_mtime < org_path.stat().st_mtime


class BuildParser(build):
    def run(self):

        parser_path = join('named_einsum', 'lark_parser.py')
        grammar_path = join('named_einsum', 'grammar.g')

        if check_artifact_out_of_date(parser_path, grammar_path):
            print('Parser out of date, rebuilding...')
            with open(parser_path, 'w') as f:
                print('Building parser...')
                code = subprocess.run([sys.executable, '-m', 'lark.tools.standalone', grammar_path], stdout=f).returncode
                if code != 0:
                    Path(parser_path).unlink(missing_ok=True)
                    raise RuntimeError('Failed to build parser')
        else:
            print('Parser is up to date!')
        super().run()


def clean_scheme(version):
    return get_local_dirty_tag(version) if version.dirty else '+clean'


setup(
    name='named_einsum',
    author='Nicolas Nytko',
    author_email='nnytko2@illinois.edu',
    use_scm_version={
        'local_scheme': clean_scheme
    },
    description='Readable einsum',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nicknytko/named_einsum',
    license='MIT',
    keywords=['einsum', 'tensor contraction'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    packages=find_packages(),
    install_requires=[
        'numpy>=1.7.0',
        'lark',
    ],
    python_requires='>=3.6',
    zip_safe=True,
    include_package_data=True,
    cmdclass={
        'build': BuildParser,
    }
)
