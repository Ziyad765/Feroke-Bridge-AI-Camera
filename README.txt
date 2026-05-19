# Exported dataset for ziyad1515 / cam

To import this data into a new Edge Impulse project, either use:

* The Edge Impulse CLI (https://docs.edgeimpulse.com/docs/edge-impulse-cli/cli-overview), run with:

    edge-impulse-uploader --clean --info-file info.labels



streamlit run streamlit_app.py --server.port 8501




* Or, via the Edge Impulse Studio. Go to **Data acquisition > Upload data**.
ok this is pefect now i want to do one thing i have a folder which have 1000 images which are not labeled i want to lable those images with this model we have created and later i want to update this model with including that dataset . 



PS C:\Users\Ziyad\Downloads\cam-export> cd 'c:\Users\Ziyad\Downloads\cam-export'
PS C:\Users\Ziyad\Downloads\cam-export> python -m pip install --force-reinstall numpy ultralytics
Collecting numpy
  Using cached numpy-2.4.3-cp312-cp312-win_amd64.whl.metadata (6.6 kB)
Collecting ultralytics
  Using cached ultralytics-8.4.24-py3-none-any.whl.metadata (39 kB)
Collecting matplotlib>=3.3.0 (from ultralytics)
  Downloading matplotlib-3.10.8-cp312-cp312-win_amd64.whl.metadata (52 kB)
Collecting opencv-python>=4.6.0 (from ultralytics)
  Using cached opencv_python-4.13.0.92-cp37-abi3-win_amd64.whl.metadata (20 kB)
Collecting pillow>=7.1.2 (from ultralytics)
  Downloading pillow-12.1.1-cp312-cp312-win_amd64.whl.metadata (9.0 kB)
Collecting pyyaml>=5.3.1 (from ultralytics)
  Downloading pyyaml-6.0.3-cp312-cp312-win_amd64.whl.metadata (2.4 kB)
Collecting requests>=2.23.0 (from ultralytics)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Collecting scipy>=1.4.1 (from ultralytics)
  Downloading scipy-1.17.1-cp312-cp312-win_amd64.whl.metadata (60 kB)
Collecting torch>=1.8.0 (from ultralytics)
  Using cached torch-2.10.0-cp312-cp312-win_amd64.whl.metadata (31 kB)
Collecting torchvision>=0.9.0 (from ultralytics)
  Using cached torchvision-0.25.0-cp312-cp312-win_amd64.whl.metadata (5.4 kB)
Collecting psutil>=5.8.0 (from ultralytics)
  Downloading psutil-7.2.2-cp37-abi3-win_amd64.whl.metadata (22 kB)
Collecting polars>=0.20.0 (from ultralytics)
  Downloading polars-1.39.3-py3-none-any.whl.metadata (10 kB)
Collecting ultralytics-thop>=2.0.18 (from ultralytics)
  Using cached ultralytics_thop-2.0.18-py3-none-any.whl.metadata (14 kB)
Collecting contourpy>=1.0.1 (from matplotlib>=3.3.0->ultralytics)
  Downloading contourpy-1.3.3-cp312-cp312-win_amd64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib>=3.3.0->ultralytics)
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib>=3.3.0->ultralytics)
  Downloading fonttools-4.62.1-cp312-cp312-win_amd64.whl.metadata (119 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib>=3.3.0->ultralytics)
  Downloading kiwisolver-1.5.0-cp312-cp312-win_amd64.whl.metadata (5.2 kB)
Collecting packaging>=20.0 (from matplotlib>=3.3.0->ultralytics)
  Using cached packaging-26.0-py3-none-any.whl.metadata (3.3 kB)
Collecting pyparsing>=3 (from matplotlib>=3.3.0->ultralytics)
  Downloading pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
Collecting python-dateutil>=2.7 (from matplotlib>=3.3.0->ultralytics)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting polars-runtime-32==1.39.3 (from polars>=0.20.0->ultralytics)
  Downloading polars_runtime_32-1.39.3-cp310-abi3-win_amd64.whl.metadata (1.5 kB)
Collecting six>=1.5 (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting charset_normalizer<4,>=2 (from requests>=2.23.0->ultralytics)
  Downloading charset_normalizer-3.4.6-cp312-cp312-win_amd64.whl.metadata (41 kB)
Collecting idna<4,>=2.5 (from requests>=2.23.0->ultralytics)
  Downloading idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests>=2.23.0->ultralytics)
  Downloading urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2017.4.17 (from requests>=2.23.0->ultralytics)
  Downloading certifi-2026.2.25-py3-none-any.whl.metadata (2.5 kB)
Collecting filelock (from torch>=1.8.0->ultralytics)
  Downloading filelock-3.25.2-py3-none-any.whl.metadata (2.0 kB)
Collecting typing-extensions>=4.10.0 (from torch>=1.8.0->ultralytics)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting sympy>=1.13.3 (from torch>=1.8.0->ultralytics)
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Collecting networkx>=2.5.1 (from torch>=1.8.0->ultralytics)
  Downloading networkx-3.6.1-py3-none-any.whl.metadata (6.8 kB)
Collecting jinja2 (from torch>=1.8.0->ultralytics)
  Using cached jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
Collecting fsspec>=0.8.5 (from torch>=1.8.0->ultralytics)
  Downloading fsspec-2026.2.0-py3-none-any.whl.metadata (10 kB)
Collecting setuptools (from torch>=1.8.0->ultralytics)
  Using cached setuptools-82.0.1-py3-none-any.whl.metadata (6.5 kB)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch>=1.8.0->ultralytics)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Collecting MarkupSafe>=2.0 (from jinja2->torch>=1.8.0->ultralytics)
  Downloading markupsafe-3.0.3-cp312-cp312-win_amd64.whl.metadata (2.8 kB)
Using cached numpy-2.4.3-cp312-cp312-win_amd64.whl (12.3 MB)
Using cached ultralytics-8.4.24-py3-none-any.whl (1.2 MB)
Downloading matplotlib-3.10.8-cp312-cp312-win_amd64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 11.0 MB/s eta 0:00:00        
Downloading contourpy-1.3.3-cp312-cp312-win_amd64.whl (226 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.62.1-cp312-cp312-win_amd64.whl (2.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 11.1 MB/s eta 0:00:00        
Downloading kiwisolver-1.5.0-cp312-cp312-win_amd64.whl (73 kB)
Using cached opencv_python-4.13.0.92-cp37-abi3-win_amd64.whl (40.2 MB)
Using cached packaging-26.0-py3-none-any.whl (74 kB)
Downloading pillow-12.1.1-cp312-cp312-win_amd64.whl (7.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.0/7.0 MB 11.4 MB/s eta 0:00:00        
Downloading polars-1.39.3-py3-none-any.whl (823 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 824.0/824.0 kB 8.9 MB/s eta 0:00:00     
Downloading polars_runtime_32-1.39.3-cp310-abi3-win_amd64.whl (47.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 47.0/47.0 MB 11.2 MB/s eta 0:00:00      
Downloading psutil-7.2.2-cp37-abi3-win_amd64.whl (137 kB)
Downloading pyparsing-3.3.2-py3-none-any.whl (122 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading pyyaml-6.0.3-cp312-cp312-win_amd64.whl (154 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.6-cp312-cp312-win_amd64.whl (154 kB)
Downloading idna-3.11-py3-none-any.whl (71 kB)
Downloading urllib3-2.6.3-py3-none-any.whl (131 kB)
Downloading certifi-2026.2.25-py3-none-any.whl (153 kB)
Downloading scipy-1.17.1-cp312-cp312-win_amd64.whl (36.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36.5/36.5 MB 11.4 MB/s eta 0:00:00      
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached torch-2.10.0-cp312-cp312-win_amd64.whl (113.8 MB)
Downloading fsspec-2026.2.0-py3-none-any.whl (202 kB)
Downloading networkx-3.6.1-py3-none-any.whl (2.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 11.6 MB/s eta 0:00:00        
Using cached sympy-1.14.0-py3-none-any.whl (6.3 MB)
Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Using cached torchvision-0.25.0-cp312-cp312-win_amd64.whl (4.3 MB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached ultralytics_thop-2.0.18-py3-none-any.whl (28 kB)
Downloading filelock-3.25.2-py3-none-any.whl (26 kB)
Using cached jinja2-3.1.6-py3-none-any.whl (134 kB)
Downloading markupsafe-3.0.3-cp312-cp312-win_amd64.whl (15 kB)
Using cached setuptools-82.0.1-py3-none-any.whl (1.0 MB)
Installing collected packages: mpmath, urllib3, typing-extensions, sympy, six, setuptools, pyyaml, pyparsing, psutil, polars-runtime-32, pillow, packaging, numpy, networkx, MarkupSafe, kiwisolver, idna, fsspec, fonttools, filelock, cycler, charset_normalizer, certifi, scipy, requests, python-dateutil, polars, opencv-python, jinja2, contourpy, torch, matplotlib, ultralytics-thop, torchvision, ultralytics
  Attempting uninstall: mpmath
    Found existing installation: mpmath 1.3.0
    Uninstalling mpmath-1.3.0:
      Successfully uninstalled mpmath-1.3.0
  Attempting uninstall: urllib3
    Found existing installation: urllib3 1.26.20
    Uninstalling urllib3-1.26.20:
      Successfully uninstalled urllib3-1.26.20
  Attempting uninstall: typing-extensions
    Found existing installation: typing_extensions 4.12.2
    Uninstalling typing_extensions-4.12.2:
      Successfully uninstalled typing_extensions-4.12.2
  Attempting uninstall: sympy
    Found existing installation: sympy 1.13.1
    Uninstalling sympy-1.13.1:
      Successfully uninstalled sympy-1.13.1
  Attempting uninstall: six
    Found existing installation: six 1.17.0
    Uninstalling six-1.17.0:
      Successfully uninstalled six-1.17.0
  Attempting uninstall: setuptools
    Found existing installation: setuptools 80.3.1
    Uninstalling setuptools-80.3.1:
      Successfully uninstalled setuptools-80.3.1
  Attempting uninstall: pyyaml
    Found existing installation: PyYAML 6.0.1
    Uninstalling PyYAML-6.0.1:
      Successfully uninstalled PyYAML-6.0.1
  Attempting uninstall: pyparsing
    Found existing installation: pyparsing 3.2.3
    Uninstalling pyparsing-3.2.3:
      Successfully uninstalled pyparsing-3.2.3
  Attempting uninstall: psutil
    Found existing installation: psutil 7.0.0
    Uninstalling psutil-7.0.0:
      Successfully uninstalled psutil-7.0.0
  Attempting uninstall: polars-runtime-32
    Found existing installation: polars-runtime-32 1.38.1
    Uninstalling polars-runtime-32-1.38.1:
      Successfully uninstalled polars-runtime-32-1.38.1
  Attempting uninstall: pillow
    Found existing installation: pillow 12.0.0
    Uninstalling pillow-12.0.0:
      Successfully uninstalled pillow-12.0.0
  Attempting uninstall: packaging
    Found existing installation: packaging 24.2
    Uninstalling packaging-24.2:
      Successfully uninstalled packaging-24.2
  Attempting uninstall: numpy
    Found existing installation: numpy 2.4.3
    Uninstalling numpy-2.4.3:
      Successfully uninstalled numpy-2.4.3
  Attempting uninstall: networkx
    Found existing installation: networkx 3.4.2
    Uninstalling networkx-3.4.2:
      Successfully uninstalled networkx-3.4.2
  Attempting uninstall: MarkupSafe
    Found existing installation: MarkupSafe 3.0.2
    Uninstalling MarkupSafe-3.0.2:
      Successfully uninstalled MarkupSafe-3.0.2
  Attempting uninstall: kiwisolver
    Found existing installation: kiwisolver 1.4.8
    Uninstalling kiwisolver-1.4.8:
      Successfully uninstalled kiwisolver-1.4.8
  Attempting uninstall: idna
    Found existing installation: idna 3.7
    Uninstalling idna-3.7:
      Successfully uninstalled idna-3.7
  Attempting uninstall: fsspec
    Found existing installation: fsspec 2024.12.0
    Uninstalling fsspec-2024.12.0:
      Successfully uninstalled fsspec-2024.12.0
  Attempting uninstall: fonttools
    Found existing installation: fonttools 4.57.0
    Uninstalling fonttools-4.57.0:
      Successfully uninstalled fonttools-4.57.0
  Attempting uninstall: filelock
    Found existing installation: filelock 3.16.1
    Uninstalling filelock-3.16.1:
      Successfully uninstalled filelock-3.16.1
  Attempting uninstall: cycler
    Found existing installation: cycler 0.12.1
    Uninstalling cycler-0.12.1:
      Successfully uninstalled cycler-0.12.1
  Attempting uninstall: charset_normalizer
    Found existing installation: charset-normalizer 3.4.1
    Uninstalling charset-normalizer-3.4.1:
      Successfully uninstalled charset-normalizer-3.4.1
  Attempting uninstall: certifi
    Found existing installation: certifi 2024.12.14
    Uninstalling certifi-2024.12.14:
      Successfully uninstalled certifi-2024.12.14
  Attempting uninstall: scipy
    Found existing installation: scipy 1.15.1
    Uninstalling scipy-1.15.1:
      Successfully uninstalled scipy-1.15.1
  Attempting uninstall: requests
    Found existing installation: requests 2.31.0
    Uninstalling requests-2.31.0:
      Successfully uninstalled requests-2.31.0
  Attempting uninstall: python-dateutil
    Found existing installation: python-dateutil 2.9.0.post0
    Uninstalling python-dateutil-2.9.0.post0:
      Successfully uninstalled python-dateutil-2.9.0.post0
  Attempting uninstall: polars
    Found existing installation: polars 1.38.1
    Uninstalling polars-1.38.1:
      Successfully uninstalled polars-1.38.1
  Attempting uninstall: opencv-python
    Found existing installation: opencv-python 4.13.0.92
    Uninstalling opencv-python-4.13.0.92:
      Successfully uninstalled opencv-python-4.13.0.92
  Attempting uninstall: jinja2
    Found existing installation: Jinja2 3.1.6
    Uninstalling Jinja2-3.1.6:
      Successfully uninstalled Jinja2-3.1.6
  Attempting uninstall: contourpy
    Found existing installation: contourpy 1.3.2
    Uninstalling contourpy-1.3.2:
      Successfully uninstalled contourpy-1.3.2
  Attempting uninstall: torch
    Found existing installation: torch 2.5.1+cu121
    Uninstalling torch-2.5.1+cu121:
      Successfully uninstalled torch-2.5.1+cu121
  Attempting uninstall: matplotlib
    Found existing installation: matplotlib 3.7.1
    Uninstalling matplotlib-3.7.1:
      Successfully uninstalled matplotlib-3.7.1
  Attempting uninstall: ultralytics-thop
    Found existing installation: ultralytics-thop 2.0.18
    Uninstalling ultralytics-thop-2.0.18:
      Successfully uninstalled ultralytics-thop-2.0.18
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.20.1+cu121
    Uninstalling torchvision-0.20.1+cu121:
      Successfully uninstalled torchvision-0.20.1+cu121
  Attempting uninstall: ultralytics
    Found existing installation: ultralytics 8.4.24
    Uninstalling ultralytics-8.4.24:
      Successfully uninstalled ultralytics-8.4.24
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
mediapipe 0.10.21 requires opencv-contrib-python, which is not installed.
roboflow 1.2.11 requires opencv-python-headless==4.10.0.84, which is not installed. 
datasets 3.6.0 requires fsspec[http]<=2025.3.0,>=2023.1.0, but you have fsspec 2026.2.0 which is incompatible.
datasets 3.6.0 requires tqdm>=4.66.3, but you have tqdm 4.66.1 which is incompatible.
email-spam-detection 1.0.0 requires click==8.1.7, but you have click 8.3.1 which is incompatible.
email-spam-detection 1.0.0 requires google-auth==2.23.4, but you have google-auth 2.48.0 which is incompatible.
email-spam-detection 1.0.0 requires pyyaml==6.0.1, but you have pyyaml 6.0.3 which is incompatible.
email-spam-detection 1.0.0 requires requests==2.31.0, but you have requests 2.32.5 which is incompatible.
gtts 2.3.2 requires click<8.2,>=7.1, but you have click 8.3.1 which is incompatible.
jupyter-console 6.6.3 requires prompt-toolkit>=3.0.30, but you have prompt-toolkit 1.0.14 which is incompatible.
mediapipe 0.10.21 requires numpy<2, but you have numpy 2.4.3 which is incompatible. 
moviepy 2.2.1 requires pillow<12.0,>=9.2.0, but you have pillow 12.1.1 which is incompatible.
numba 0.61.2 requires numpy<2.3,>=1.24, but you have numpy 2.4.3 which is incompatible.
roboflow 1.2.11 requires idna==3.7, but you have idna 3.11 which is incompatible.   
tokenizers 0.21.0 requires huggingface-hub<1.0,>=0.16.4, but you have huggingface-hub 1.7.2 which is incompatible.
torchaudio 2.5.1 requires torch==2.5.1, but you have torch 2.10.0 which is incompatible.
transformers 4.48.1 requires huggingface-hub<1.0,>=0.24.0, but you have huggingface-hub 1.7.2 which is incompatible.
Successfully installed MarkupSafe-3.0.3 certifi-2026.2.25 charset_normalizer-3.4.6 contourpy-1.3.3 cycler-0.12.1 filelock-3.25.2 fonttools-4.62.1 fsspec-2026.2.0 idna-3.11 jinja2-3.1.6 kiwisolver-1.5.0 matplotlib-3.10.8 mpmath-1.3.0 networkx-3.6.1 numpy-2.4.3 opencv-python-4.13.0.92 packaging-26.0 pillow-12.1.1 polars-1.39.3 polars-runtime-32-1.39.3 psutil-7.2.2 pyparsing-3.3.2 python-dateutil-2.9.0.post0 pyyaml-6.0.3 requests-2.32.5 scipy-1.17.1 setuptools-82.0.1 six-1.17.0 sympy-1.14.0 torch-2.10.0 torchvision-0.25.0 typing-extensions-4.15.0 ultralytics-8.4.24 ultralytics-thop-2.0.18 urllib3-2.6.3
