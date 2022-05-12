from setuptools import setup
import glob
import os
import sys
import json

sys.path.append("hera_filters")
import version

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_filters', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

data_files = package_files('hera_filters', 'data')

setup_args = {
    'name': 'hera_filters',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/hera_filters',
    'license': 'MIT',
    'description': 'Tools useful for the handling, visualization, and analysis of interferometric data.',
    'package_dir': {'hera_filters': 'hera_filters'},
    'packages': ['hera_filters'],
    'package_data': {'hera_filters': data_files},
    'version': version.version,
    'include_package_data': True,
    'scripts': glob.glob('scripts/*'),
    'install_requires':[
        'numpy',
        'six',
        'scipy',
    ],
    'extras_require': {'aipy':['aipy>=3.0rc2']}
}


if __name__ == '__main__':
    setup(**setup_args)
