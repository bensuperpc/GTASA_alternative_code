import os
import json
import yaml
from pathlib import Path
import subprocess
import sys
import time
import multiprocessing
from joblib import Parallel, delayed
sys.stdout.reconfigure(encoding='utf-8')

def build_docker(docker_image=None, command=None):
    start = time.process_time()
    pipe = subprocess.Popen(['docker run --rm -v "$PWD":/usr/src/myapp -w /usr/src/myapp ' +
                            docker_image + ' ' + command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res = pipe.communicate()
    print((time.process_time() - start)*1000)
    print("retcode: ", pipe.returncode)
    print("stdout :\n" + res[0].decode("utf-8"))
    print("stderr :\n" + res[1].decode("utf-8"))


data = []
with open('data/data.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    
#with open('data/data2.yaml', 'w') as outfile:
#    yaml.dump(data, outfile, default_style=None, default_flow_style=False)

commands = list()

BaseArch = data['DockerCompiler'][0]['BaseArch']

for compiler in data['DockerCompiler'][1]['Compilers']:
    for arch in compiler['ArchSupport']:
        for opti in compiler['OptimisationFlags']:
            DockerImage = str(compiler['DockerImage']) + \
                ':' + str(compiler['DockerTag'])
            print('DockerImage:', DockerImage)
            print('Arch:', arch)
            print('Opti:', opti)
            print('CompilerCommand:', compiler['CompilerCommand'])
            print('CPPStandard:', compiler['CPPStandard'])

            CFLAGS = '-fopenmp -m64 -' + opti + ' -march=' + arch + ' -std=' + compiler['CPPStandard']
            TARGET ='gta' + '_' + compiler['CompilerCommand'] + '_' + str(compiler['DockerTag']) + '_' + arch + '_' + opti
            CC = compiler['CompilerCommand']

            Command = 'make all ' + 'CFLAGS="' + CFLAGS + '"' + ' TARGET=' + TARGET + ' CC=' + CC
            print('Command:', Command)
            commands.append([DockerImage, Command])
            #build_docker(DockerImage, Command)
print(len(commands))

num_cores = multiprocessing.cpu_count()
print(num_cores)
processed_list = Parallel(n_jobs=num_cores)(delayed(wrapper)(build_docker, command[0], command[1]) for command in commands)
